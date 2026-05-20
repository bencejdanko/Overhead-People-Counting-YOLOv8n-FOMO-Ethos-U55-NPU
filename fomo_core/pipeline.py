import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

def generate_gaussian_target(grid, center_x, center_y, grid_size=24, sigma=0.8):
    y_coords, x_coords = np.ogrid[:grid_size, :grid_size]
    dist_sq = (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
    gaussian_smear = np.exp(-dist_sq / (2 * sigma ** 2))
    grid[:, :, 0] = np.maximum(grid[:, :, 0], gaussian_smear)
    return grid

class FomoDatasetBuilder:
    def __init__(self, annotator, input_size=192, grid_size=24):
        self.annotator = annotator
        self.input_size = input_size
        self.grid_size = grid_size

    def process_example(self, example):
        img = example['image'].convert('RGB')
        true_centers = self.annotator.get_true_centroids(img, example_objects=example['objects'])
        
        img_resized = img.resize((self.input_size, self.input_size))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = preprocess_input(img_array)
        
        grid = np.zeros((self.grid_size, self.grid_size, 1), dtype=np.float32)
        for x_norm, y_norm in true_centers:
            grid_x = int(x_norm * self.grid_size)
            grid_y = int(y_norm * self.grid_size)
            grid_x = max(0, min(self.grid_size - 1, grid_x))
            grid_y = max(0, min(self.grid_size - 1, grid_y))
            grid = generate_gaussian_target(grid, grid_x, grid_y, self.grid_size)
            
        return img_array, grid

    @staticmethod
    @tf.function
    def synchronized_augment(image, target):
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_left_right(image)
            target = tf.image.flip_left_right(target)
        if tf.random.uniform([]) > 0.5:
            image = tf.image.flip_up_down(image)
            target = tf.image.flip_up_down(target)
            
        rot_k = tf.random.uniform([], minval=0, maxval=4, dtype=tf.int32)
        image = tf.image.rot90(image, k=rot_k)
        target = tf.image.rot90(target, k=rot_k)
        return image, target