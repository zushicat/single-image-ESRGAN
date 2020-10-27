import os

from data_loader import DataLoader

import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras


MODEL = "../model/image_generator_model.h5"

# ****
# some of my controle images
# ***
FILE_INPUT_PATH = "/Users/karin/programming/data/ortho-images/default_test_images/lr"
# FILE_INPUT_PATH = "/Users/karin/programming/data/ortho-images/ortho_400/2013_altstadt_nord"

FILE_OUTPUT_PATH = "../test_predictions/model_predictions"

IMAGE_IN_SIZE = 400
IMAGE_OUT_SIZE = 600


if __name__ == "__main__":
    def create_img_dir():
        save_img_dir = f"{FILE_OUTPUT_PATH}/{IMAGE_IN_SIZE}_{IMAGE_OUT_SIZE}"

        if not os.path.exists(f"{save_img_dir}/in_{IMAGE_IN_SIZE}"):
            os.makedirs(f"{save_img_dir}/in_{IMAGE_IN_SIZE}")
        if not os.path.exists(f"{save_img_dir}/out_{IMAGE_OUT_SIZE}"):
            os.makedirs(f"{save_img_dir}/out_{IMAGE_OUT_SIZE}")

        return save_img_dir
    

    def resolve_single_image(file_name):
        input_image = data_loader.load_single_image(f"{FILE_INPUT_PATH}/{file_name}", IMAGE_IN_SIZE)
        
        output_image = model.predict(input_image)  # predict image
       
        # ***
        # for both in/out image: de-normalize image colors and change data format
        # ***
        input_image = 0.5 * input_image + 0.5
        input_image = Image.fromarray((np.uint8(input_image*255)[0]))
        
        output_image = 0.5 * output_image + 0.5
        output_image = Image.fromarray((np.uint8(output_image*255)[0]))
        
        # ***
        # resize out image
        # ***
        output_image = output_image.resize((IMAGE_OUT_SIZE, IMAGE_OUT_SIZE), Image.BICUBIC)
        
        return input_image, output_image

    # load the trained model (generator)
    # Regarding compile=False: https://stackoverflow.com/a/57054106
    model = keras.models.load_model(MODEL, compile=False)

    data_loader = DataLoader()  # makes your life easier 
    save_img_dir = create_img_dir()

    # ****
    # some of my controle images
    # ***
    file_names = [
        "354400_5643700_354500_5643800.png",
        "354600_5643700_354700_5643800.png",
        "354900_5642600_355000_5642700.png",
        "355100_5643700_355200_5643800.png",
        "355300_5642700_355400_5642800.png"
    ]

    # file_names = [
    #     "355174_5644507_355274_5644607.png"
    # ]

    # ***
    # predict each image
    # ***
    for file_name in file_names:
        input_image, output_image = resolve_single_image(file_name)
        
        # ***
        # save both images in their respective dirs
        # ***
        input_image.save(f"{save_img_dir}/in_{IMAGE_IN_SIZE}/{file_name}")
        output_image.save(f"{save_img_dir}/out_{IMAGE_OUT_SIZE}/{file_name}")
