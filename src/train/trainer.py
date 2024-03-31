import os
import torch
import numpy as np
from datetime import datetime
from src.train.loss import build_loss
from src.logger.loggers import AbstractLogger
from src.train.optimizer import build_optimizer
from src.utils.dict_as_member import DictAsMember
from src.evalutation.evaluators import AbstractEvaluator
from src.train.lr_scheduler import build_lr_scheduler
from src.utils.dict_as_member import DictAsMember


class Trainer:
    def __init__(self, 
                 model: torch.nn.Module, 
                 logger: AbstractLogger, 
                 train_evaluator: AbstractEvaluator, 
                 valid_evaluator: AbstractEvaluator, 
                 config: DictAsMember) -> None:
        self.model = model
        self.logger = logger
        self.config = config.train
        self.train_evaluator = train_evaluator
        self.valid_evaluator = valid_evaluator
        self.optimizer = build_optimizer(model.parameters(), self.config)
        self.loss = build_loss(self.config)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, self.config)
        self.training_loss = []
        self.validation_loss = []
        self.actual_lr = self.config.optimizer_config.lr
        
        if config.train.save_model:
            self.best_model_path = os.path.join(self.config.out_path, config.name + '_best.pth')
            if not os.path.exists(self.config.out_path):
                os.makedirs(self.config.out_path)

    def train(self, trainloader, validloader):
        self.start_time = datetime.now()
        try:
            self.n_trained_batches = 0
            self.batches_per_epoch = len(trainloader)
            self.record_loss = float('inf')

            self.model.train()
            for epoch in range(self.config.n_epochs): # Epoch loop
                trainiter = iter(trainloader)
                running_loss = 0.
                for i in range(self.batches_per_epoch): # Batch loop
                    running_loss += self.train_step(trainiter)
                    self.n_trained_batches += 1

                    if self.n_trained_batches % self.config.batches_per_evaluation == 0:
                        running_loss /= self.config.batches_per_evaluation
                        self.training_loss.append(running_loss)

                        stop = self.evaluation(trainloader, validloader)
                        if stop:
                            break
                        else:
                            self.model.train()
                if stop:
                    break
                self.logger.log_bold(f'Finished epoch {epoch + 1}')
        except KeyboardInterrupt:
            self.logger.log_warning("Training interrupted at epoch:", epoch)
            stop = True
        
        self.train_evaluator.flush()
        self.valid_evaluator.flush()
        self.logger.log_header("Training finished")
        self.load_best_weights()
        self.print_final_info()

    def train_step(self, trainiter):
        x, y = next(trainiter)
        x, y = x.to(self.model.device), y.to(self.model.device)
        self.optimizer.zero_grad()
        out = self.model.forward(x)
        loss = self.loss(out, y)
        loss_value = loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), 
            self.config.max_grad_norm, 
            norm_type=self.config.grad_norm_type
        )
        self.optimizer.step()
        return loss_value
    
    def evaluation(self, trainloader, validloader):
        self.model.eval()
        epoch_valid_loss = self.evaluate_model(trainloader, validloader)
        self.validation_loss.append(epoch_valid_loss)
        self.logger.log("Evaluation at", self.n_trained_batches + 1, "batches results")
        self.logger.log("Train loss:", self.training_loss[-1])
        self.logger.log("Valid loss:", epoch_valid_loss)

        stop, saved = self.early_stopping(epoch_valid_loss)
        if saved:
            self.logger.log_success("New valid loss record, model saved!")
        if stop:
            self.logger.log_warning("Early stopped")

        self.lr_scheduler.step(epoch_valid_loss)
        lr = self.lr_scheduler.get_last_lr()[0]
        if lr != self.actual_lr:
            self.logger.log_warning("Lr reduced to:", lr)
            self.actual_lr = lr
            
        return stop

    def evaluate_model(self, trainloader, validloader):
        epoch_valid_loss = 0.

        with torch.no_grad():
            for i, (x, y) in enumerate(validloader):
                x, y = x.to(self.model.device), y.to(self.model.device)
                out = self.model.forward(x)
                loss = self.loss(out, y)
                epoch_valid_loss += loss.item()

                self.valid_evaluator.evaluate(out, y, i/len(validloader))
            self.valid_evaluator.write(self.n_trained_batches)

            for i, (x, y) in enumerate(trainloader):
                x, y = x.to(self.model.device), y.to(self.model.device)
                out = self.model.forward(x)
                self.train_evaluator.evaluate(out, y, i/len(trainloader))
            self.train_evaluator.write(self.n_trained_batches)

        return epoch_valid_loss / len(validloader)

    def early_stopping(self, valid_loss):
        saved = False
        if valid_loss < self.record_loss:
            self.record_loss = valid_loss
            if self.config.save_model:
                torch.save(self.model.state_dict(), self.best_model_path)
                self.model_saved_batch = self.n_trained_batches
                saved = True
            self.epochs_since_loss_record = 0
        else:
            self.epochs_since_loss_record += 1

        if self.epochs_since_loss_record > self.config.early_stopping_patience:
            return True, saved
        return False, saved

    def load_best_weights(self):
        if self.config.save_model:
            state_dict = torch.load(self.best_model_path)
            self.model.load_state_dict(state_dict)
            self.logger.log("Loaded best model weights!")

    def print_final_info(self):
        self.logger.log("Total training time:", str(datetime.now() - self.start_time))
        self.logger.log("Trained for", self.n_trained_batches, "batches,", np.ceil(self.n_trained_batches / self.batches_per_epoch), "epochs")
        self.logger.log("Final training loss:", self.training_loss[-1])
        self.logger.log("Final validation loss:", self.validation_loss[-1])
    
