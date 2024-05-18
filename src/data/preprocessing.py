# This code is used to preprocess the data before training the model.
# The data is made up of images and labels. The images are loaded from the disk and the labels are read from a CSV file.
# Images are preprocessed using the python library openCV.

import numpy as np
from PIL import Image
import pandas as pd


def preprocessing():
    data = pd.read_csv('./data/labels.csv', sep=',')
    patients = np.unique(data['Patient ID'])

    im_list = []

    # Preprocess the data according to the machine used for obtaining the images.
    for i in range(len(data["Anonymized ID"])):
        if data["machine"][i] == "samsung" :

            # Samsung images are very clear. Only a crop is needed.

            id = data["Anonymized ID"][i]
            im = Image.open(f'./data/imgs/{id:03}.jpg')
            width, height = im.size
            left = width * 0.1
            upper = height * 0.05
            right = width * .9
            lower = height * .7

            crop_area = (left, upper, right, lower)

            # Crop the image
            cropped_image = im.crop(crop_area)
            label = data["OMERACT score"][i]
            im_list.append((cropped_image, label, "samsung"))

        elif data["machine"][i] == "philips" :  

            # Phillips images ..............

            id = data["Anonymized ID"][i]
            im = Image.open(f'./data/imgs/{id:03}.jpg')
            width, height = im.size
            left = width * 0.15
            upper = height * 0.01
            right = width * .85
            lower = height * .7

            crop_area = (left, upper, right, lower)

            # Crop the image
            cropped_image = im.crop(crop_area)
            label = data["OMERACT score"][i]
            im_list.append((cropped_image, label, "philips")) 


        elif data["machine"][i] == "esaote" :  

            # Esaote images .............. will probably need more preprocessing

            id = data["Anonymized ID"][i]
            im = Image.open(f'./data/imgs/{id:03}.jpg')
            width, height = im.size
            left = width * 0.10
            upper = height * 0.01
            right = width * .9
            lower = height * .8

            crop_area = (left, upper, right, lower)

            # Crop the image
            cropped_image = im.crop(crop_area)
            label = data["OMERACT score"][i]
            im_list.append((cropped_image, label, "esaote")) 

        elif data["machine"][i] == "GE" :  

            # GE images .............. will probably need more preprocessing

            id = data["Anonymized ID"][i]
            im = Image.open(f'./data/imgs/{id:03}.jpg')

            cropped_image = im
            
            label = data["OMERACT score"][i]
            im_list.append((cropped_image, label, "GE")) 

    for i in range(len(im_list)):
        id = data["Anonymized ID"][i]
        # Save the images in the folder preprocessed_images
        im_list[i][0].save(f'./data/preprocessed_images/{id:03}.jpg')

    return 