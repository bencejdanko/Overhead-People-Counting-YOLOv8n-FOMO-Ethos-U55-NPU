from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BACKGROUND_CLASS = 0
OBJECT_CLASS = 1


def set_discrete_target(grid, center_x, center_y, grid_size=24):
    grid_x = int(center_x * grid_size)
    grid_y = int(center_y * grid_size)
    grid_x = max(0, min(grid_size - 1, grid_x))
    grid_y = max(0, min(grid_size - 1, grid_y))
    grid[grid_y, grid_x, BACKGROUND_CLASS] = 0.0
    grid[grid_y, grid_x, OBJECT_CLASS] = 1.0
    return grid

class FomoDatasetBuilder:
    def __init__(self, annotator, input_size=192, grid_size=24, num_classes=2):
        self.annotator = annotator
        self.input_size = input_size
        self.grid_size = grid_size
        self.num_classes = num_classes

    def process_example(self, example):
        img = load_pil_image(example['image'])
        true_centers = self.annotator.get_true_centroids(img, example_objects=example['objects'])
        
        img_resized = img.resize((self.input_size, self.input_size))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = preprocess_input(img_array)
        
        grid = np.zeros((self.grid_size, self.grid_size, self.num_classes), dtype=np.float32)
        grid[:, :, BACKGROUND_CLASS] = 1.0
        for x_norm, y_norm in true_centers:
            grid = set_discrete_target(grid, x_norm, y_norm, self.grid_size)
            
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


def load_pil_image(image_value):
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            return Image.open(BytesIO(image_value["bytes"])).convert("RGB")
        if image_value.get("path"):
            return Image.open(image_value["path"]).convert("RGB")

    raise TypeError(f"Unsupported image value: {type(image_value)!r}")


def build_arrays_from_hf_dataset(builder, hf_dataset, split_name, num_workers=8):
    print(f"Extracting {split_name} features...")
    examples = list(hf_dataset)

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        processed = list(executor.map(builder.process_example, examples))

    images, targets = zip(*processed)
    return np.stack(images), np.stack(targets)


def iter_processed_hf_dataset(builder, hf_dataset):
    for example in hf_dataset:
        yield builder.process_example(example)


def build_tf_dataset_from_hf_dataset(builder, hf_dataset):
    output_signature = (
        tf.TensorSpec(
            shape=(builder.input_size, builder.input_size, 3),
            dtype=tf.float32,
        ),
        tf.TensorSpec(
            shape=(builder.grid_size, builder.grid_size, builder.num_classes),
            dtype=tf.float32,
        ),
    )
    return tf.data.Dataset.from_generator(
        lambda: iter_processed_hf_dataset(builder, hf_dataset),
        output_signature=output_signature,
    )


def build_tf_dataset_from_hf_datasets(builder, hf_datasets):
    datasets = [
        build_tf_dataset_from_hf_dataset(builder, hf_dataset)
        for hf_dataset in hf_datasets
    ]
    if not datasets:
        raise ValueError("At least one Hugging Face dataset is required")

    dataset = datasets[0]
    for next_dataset in datasets[1:]:
        dataset = dataset.concatenate(next_dataset)
    return dataset


def build_sample_arrays_from_hf_datasets(builder, hf_datasets, sample_count=16):
    images = []
    targets = []
    for hf_dataset in hf_datasets:
        for example in hf_dataset:
            image, target = builder.process_example(example)
            images.append(image)
            targets.append(target)
            if len(images) >= sample_count:
                return np.stack(images), np.stack(targets)

    if not images:
        raise ValueError("Could not build samples from empty datasets")

    return np.stack(images), np.stack(targets)
