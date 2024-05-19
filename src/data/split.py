from sklearn.model_selection import train_test_split
from PIL import Image
import pandas as pd
import numpy as np

def data_split(valid_size=.15, test_size=.15, random_state=42):   
    data = pd.read_csv('./data/labels.csv', sep=';')
    patients = np.unique(data['Patient ID'])
    stratify_keys = [(
        # data[data['Patient ID'] == id].iloc[0].Center,
        np.max(data[data['Patient ID'] == id]["OMERACT score"])
        ) for id in patients
    ]

    train_patients, test_patients, train_stratify_keys, _ = train_test_split(patients, stratify_keys, test_size=test_size, random_state=random_state, stratify=stratify_keys)
    train_patients, valid_patients = train_test_split(train_patients, test_size=valid_size, random_state=random_state, stratify=train_stratify_keys)

    def get_data(patients):
        ids_list = sum([
            data[data['Patient ID'] == id]['Anonymized ID'].to_list()
            for id in patients
        ], [])
        score_list = sum([
            data[data['Patient ID'] == id]['OMERACT score'].to_list()
            for id in patients
        ], [])

        imgs = [
            np.repeat(np.array(Image.open(f'./data/imgs/{id:03}.jpg'))[:, :, np.newaxis], 3, -1)
            for id in ids_list
        ]

        return imgs, np.array(score_list).astype(np.int64)

    train_data = get_data(train_patients)
    valid_data = get_data(valid_patients)
    test_data = get_data(test_patients)
    return train_data, valid_data, test_data


