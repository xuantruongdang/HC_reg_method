import os
import random
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler


# https://github.com/HasnainRaz/SemSegPipeline/blob/master/dataloader.py
AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataLoader(object):
    """
    A TensorFlow Dataset API based loader for semantic segmentation problems.
    """

    def __init__(self, root, mode="train", augmentation=False, image_size=(224, 224, 3)):
        """
        root: "../data/training_set"
        """
        super().__init__()
        self.root = root
        self.augmentation = augmentation
        self.image_size = (image_size[0], image_size[1])

        if (mode == "train"):
            self.df = pd.read_csv("./data/train.csv")
        elif (mode == "valid"):
            self.df = pd.read_csv("./data/valid.csv")
        elif (mode == "test"):
            self.df = pd.read_csv("./test_set_pixel_size.csv")

        if mode in ["train", "valid"]:
            self.parse_data_path()
        elif mode == "test":
            pass
            # self.parse_data_test

    def parse_data_path(self):
        self.image_paths = [os.path.join(self.root, row["filename"]) for i, row in self.df.iterrows()]
        print(self.image_paths)

        labels = [row[["head circumference (mm)"]].to_numpy() for i, row in self.df.iterrows()]
        print(type(labels))
        max_HC = max(labels)
        print("max HC: ", max_HC)
        self.labels = [x / max_HC for x in labels]
        print(type(self.labels))
        print('label list: ', len(self.labels))

    def parse_data(self, image_paths, labels):
        image_content = tf.io.read_file(image_paths)
        images = tf.image.decode_png(image_content, channels=1)
        images = tf.cast(images, tf.float32)

        labels = tf.convert_to_tensor(labels, dtype=tf.float32)

        return images, labels

    def crop_data(self, image, label):
        # xmin, ymin, xmax, ymax = list(map(int, label[1:]))
        h, w = image.shape[:2]
        crop = tf.image.crop_to_bounding_box(image, 0, 0, h, w)

        label = label[0]

        return image, crop, label

    def normalize_data(self, image, crop, label):
        """
        Normalize images
        """
        # scaler = StandardScaler()
        image /= 255
        # print('shape imageeeeeeeeeeeeeeeeeeeeeeeee: ', image.shape, type(image))
        # crop /= 255
        # print('shape imageeeeeeeeeeeeeeeeeeeeeeeeecrop: ', crop.shape, type(crop))
        # Mean and Std of ImagesNet
        means = [0.485, 0.456, 0.406]
        mean = sum(means) / len(means)
        stds = [0.229, 0.224, 0.225]
        std = sum(stds) / len(stds)
        image = (image - mean) / std
        # print('shape imageeeeeeeeeeeeeeeeeeeeeeeee: ', image.shape, type(image))
        crop = (crop - mean) / std

        # image = scaler.fit_transform(image)
        # crop = scaler.fit_transform(crop)

        # image = tf.convert_to_tensor(image)
        # crop = tf.convert_to_tensor(crop)
        # image = tf.image.per_image_standardization(image)

        return image, crop, label

    def resize_data(self, image, crop, label):
        """
        Resizes images to specified size.
        """

        image = tf.image.resize_with_pad(image, self.image_size[0], self.image_size[1], method="nearest")
        crop = tf.image.resize_with_pad(crop, self.image_size[0], self.image_size[1], method="nearest")

        return image, crop, label

    def grayscale_to_rgb(self, image, crop, label):
        image =  tf.image.grayscale_to_rgb(image)
        crop =  tf.image.grayscale_to_rgb(crop)

        return image, crop, label

    def change_brightness(self, image, label):
        """
        Radnomly applies a random brightness change.
        """
        cond_brightness = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_brightness, lambda: tf.image.random_brightness(
            image, 0.2), lambda: tf.identity(image))

        return image, label

    def change_contrast(self, image, label):
        """
        Randomly applies a random contrast change.
        """
        cond_contrast = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
        image = tf.cond(cond_contrast, lambda: tf.image.random_contrast(
            image, 0.1, 0.5), lambda: tf.identity(image))

        return image, label

    def flip_horizontally(self, image, label):
        """
        Randomly flips image horizontally in accord.
        """
        image = tf.image.random_flip_left_right(comb_tensor)

        return image, label

    def reorder_channel(self, tensor, order="channel_first"):
        if order == "channel_first":
            return tf.convert_to_tensor(np.moveaxis(tensor.numpy(), -1, 0))
        else:
            return tf.convert_to_tensor(np.moveaxis(tensor.numpy(), 0, -1))

    def _tranform(self, tensor, types):
        tensor = self.reorder_channel(tensor)

        if types == "rotate":
            tensor = tf.keras.preprocessing.image.random_rotation(tensor, 30)
        elif types == "shift":
            tensor = tf.keras.preprocessing.image.random_shift(
                tensor, 0.3, 0.3)
        else:
            tensor = tf.keras.preprocessing.image.random_zoom(tensor, 0.3)

        tensor = self.reorder_channel(
            tf.convert_to_tensor(tensor), order="channel_last")

        return tensor

    def rotate(self, image, label):
        cond_rotate = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
      
        image = tf.cond(cond_rotate, lambda: self._tranform(
            image, "rotate"), lambda: tf.identity(image))

        return image, label

    def shift(self, image, label):
        cond_shift = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
      
        image = tf.cond(cond_shift, lambda: self._tranform(
            image, "shift"), lambda: tf.identity(image))

        return image, label

    def zoom(self, image, label):
        cond_zoom = tf.cast(tf.random.uniform(
            [], maxval=2, dtype=tf.int32), tf.bool)
      
        image = tf.cond(cond_zoom, lambda: self._tranform(
            image, "zoom"), lambda: tf.identity(image))

        return image, label

    @tf.function
    def map_function(self, images_path, labels):
        image, label = self.parse_data(images_path, labels)

        def augmentation_func(image_f, label_f):
            image_f, crop_f, label_f = self.crop_data(image_f, label_f)
            image_f, crop_f, label_f = self.normalize_data(image_f, crop_f, label_f)

            # if self.augmentation:
            #     image_f, label_f = self.change_brightness(image_f, label_f)
            #     image_f, label_f = self.change_contrast(image_f, label_f)
            #     image_f, label_f = self.flip_horizontally(image_f, label_f)
            #     image_f, label_f = self.rotate(image_f, label_f)
            #     image_f, label_f = self.shift(image_f, label_f)

            image_f, crop_f, label_f = self.resize_data(image_f, crop_f, label_f)
            image_f, crop_f, label_f = self.grayscale_to_rgb(image_f, crop_f, label_f)

            return crop_f, label_f

        return tf.py_function(augmentation_func, [image, label], [tf.float32, tf.float32])

    def data_gen(self, batch_size, shuffle=False):
        # Create dataset out of the 2 files:
        data = tf.data.Dataset.from_tensor_slices(
            (self.image_paths, self.labels))

        # Parse images and labels
        data = data.map(self.map_function, num_parallel_calls=AUTOTUNE)

        if shuffle:
            # Prefetch, shuffle then batch
            data = data.prefetch(AUTOTUNE).shuffle(
                random.randint(0, len(self.image_paths))).batch(batch_size)
        else:
            # Batch and prefetch
            data = data.batch(batch_size).prefetch(AUTOTUNE)

        return data