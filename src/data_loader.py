'''
Get random images from path defined in BASE_PATH
- augment image
- take random crop from image (parameter "crop_size"): hr (original) image crop
- downscale by factor (parameter "scale_factor"): lr version of hr image crop
- apply on number of images (parameter "batch_size")
- normalize batch images ([0, 255] -> [-1, 1])
'''
from glob import glob
import os

import numpy as np
from PIL import Image, ImageFilter
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tqdm import tqdm


BASE_PATH = "/Users/karin/programming/data/ortho-images/ortho_2019_1600_1600"


class DataLoader():
    def __init__(self, crop_size=96, scale_factor=4, percent_of_training_set=0.5):
        self.datagen = ImageDataGenerator(
            rotation_range=90,
            horizontal_flip=True,
            vertical_flip=True,
            width_shift_range=0.1,
            height_shift_range=0.1,
            fill_mode="reflect"
        )

        self.crop_size = crop_size
        self.scale_factor = scale_factor

        self.percent_of_training_set = percent_of_training_set

        self.image_paths = glob(f'{BASE_PATH}/**/*.png', recursive=True)  # all image paths from BASE_PATH or it's subdirs
        self.images = []


    def load_images(self):
        '''
        To reduce computation time when getting augmented image crops, divide each high res image in advance
        into slightly bigger crops than used by augmentation
        '''
        dist_to_image_border = 20
        tmp_crop_size = self.crop_size * 2 + dist_to_image_border

        num_images_from_training_set = int(len(self.image_paths)*self.percent_of_training_set)
        print(f"Selected number of high res images from training set: {num_images_from_training_set} ({int(self.percent_of_training_set*100)}%)")
        
        # ***
        # get a portion of images to reduce initial computation time
        # should be sufficient if i.e. 9*9 (number of crops) * 200 (random images) = 16200 copped images
        # ***
        random_img_path_selection = np.random.choice(self.image_paths, size=num_images_from_training_set)

        # ***
        # divide high res images: get (bigger) image crops of all high res images
        # first loop with tqdm progress bar to show loading progress
        # ***
        for img_path in tqdm(random_img_path_selection, desc="Loading training set"): 
            original_img = load_img(img_path)  # type: PIL image in RGB

            for size in [None, original_img.height//2, original_img.height//4]:  # additionally scale incoming hr: 1600 -> None, 800, 400
                left = 0
                upper = 0
                right = tmp_crop_size
                lower = tmp_crop_size
                
                img = original_img

                if size is not None:
                    img = original_img.resize((size, size), Image.BICUBIC)
                
                for _ in range(img.height//tmp_crop_size):  # each row
                    for _ in range(img.width//tmp_crop_size):  # each column
                        self.images.append(img.crop((left, upper, right, lower)))

                        left += tmp_crop_size
                        right += tmp_crop_size

                    # line done
                    left = 0
                    right = tmp_crop_size

                    upper += tmp_crop_size
                    lower += tmp_crop_size

    
    def crop_image(self, img):
        dist_to_image_border = 20
        upper_left = np.random.randint(low=dist_to_image_border, high=img.width-dist_to_image_border-self.crop_size)
        lower_right = upper_left+self.crop_size
        return img.crop((upper_left, upper_left, lower_right, lower_right))


    def load_data(self, batch_size=1):
        # ***
        # initial image load
        # ***
        if len(self.images) == 0:
            self.load_images()

        # ***
        # get a batch of hr image crops
        # ***
        batch_images = []
        batch_images_indice = np.random.choice(len(self.images), size=batch_size)  # type: PIL image

        for index in batch_images_indice:
            batch_images.append(self.images[index])
        
        # ***
        # augment and store the crops of this batch
        # ***
        imgs_hr = []
        imgs_lr = []
        for img_hr in batch_images:
            # ***
            # image augmentation
            # ***
            data = np.expand_dims(img_to_array(img_hr), 0)
            it = self.datagen.flow(data, batch_size=1)
            augmented_img_np_array = it.next()[0].astype('uint8')
            
            img_hr = Image.fromarray(augmented_img_np_array)
            img_hr = self.crop_image(img_hr)
            # img_hr.show()  # debug
            
            # ***
            # create downsampled low res image (crop)
            # ***
            img_lr = img_hr.resize((self.crop_size//self.scale_factor, self.crop_size//self.scale_factor), Image.BICUBIC)
            # img_lr.show()  # debug
            
            # ***
            # store both in arrays
            # ***
            imgs_hr.append(np.asarray(img_hr))
            imgs_lr.append(np.asarray(img_lr))

        # ***
        # normalize: [0, 255] -> [-1, 1]
        # ***
        imgs_hr = np.array(imgs_hr) / 127.5 - 1.
        imgs_lr = np.array(imgs_lr) / 127.5 - 1.

        return imgs_hr, imgs_lr


    def load_single_image(self, file_path, size=None):
        img = Image.open(file_path)
        if size is not None:
            img = img.resize((size, size))
        img_np_array = [np.asarray(img)]
        return np.array(img_np_array) / 127.5 - 1.



if __name__ == "__main__":
    data_loader = DataLoader(crop_size=96, scale_factor=4, percent_of_training_set=0.01)
    imgs_hr, imgs_lr = data_loader.load_data(batch_size=1)  # default: False
    # print(imgs_lr[0].shape, imgs_hr[0].shape)
