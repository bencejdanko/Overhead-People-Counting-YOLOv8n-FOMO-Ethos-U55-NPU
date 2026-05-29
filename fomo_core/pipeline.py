from io import BytesIO
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BACKGROUND_CLASS = 0
OBJECT_CLASS = 1
DEFAULT_MAX_OBJECTS = 256


def set_discrete_target(grid, center_x, center_y, grid_size=24):
    grid_x = int(center_x * grid_size)
    grid_y = int(center_y * grid_size)
    grid_x = max(0, min(grid_size - 1, grid_x))
    grid_y = max(0, min(grid_size - 1, grid_y))
    grid[grid_y, grid_x, BACKGROUND_CLASS] = 0.0
    grid[grid_y, grid_x, OBJECT_CLASS] = 1.0
    return grid

class FomoDatasetBuilder:
    def __init__(
        self,
        annotator,
        input_size=192,
        grid_size=24,
        num_classes=2,
        center_crop_size=None,
        max_objects=DEFAULT_MAX_OBJECTS,
    ):
        self.annotator = annotator
        self.input_size = input_size
        self.grid_size = grid_size
        self.num_classes = num_classes
        self.center_crop_size = center_crop_size
        self.max_objects = max_objects

    def process_example(self, example):
        img = load_pil_image(example['image'])
        true_centers = self.annotator.get_true_centroids(img, example_objects=example['objects'])
        
        if self.center_crop_size is not None:
            img, true_centers = center_crop_pil_image_and_centroids(
                img,
                true_centers,
                self.center_crop_size,
            )

        img_resized = img.resize((self.input_size, self.input_size))
        img_array = np.array(img_resized, dtype=np.float32)
        img_array = preprocess_input(img_array)
        
        grid = np.zeros((self.grid_size, self.grid_size, self.num_classes), dtype=np.float32)
        grid[:, :, BACKGROUND_CLASS] = 1.0
        for x_norm, y_norm in true_centers:
            grid = set_discrete_target(grid, x_norm, y_norm, self.grid_size)
            
        return img_array, grid

    def process_raw_example(self, example, center_crop_size=None):
        img = load_pil_image(example["image"])
        true_centers = self.annotator.get_true_centroids(
            img,
            example_objects=example["objects"],
        )
        image = np.array(img, dtype=np.uint8)
        centroids, centroid_mask = pad_centroids(
            true_centers,
            max_objects=self.max_objects,
        )
        return image, centroids, centroid_mask, np.int32(center_crop_size or 0)

    @tf.function
    def preprocess_raw_example(self, image, centroids, centroid_mask, center_crop_size):
        image = tf.cast(image, tf.float32)
        image_shape = tf.shape(image)
        height = tf.cast(image_shape[0], tf.float32)
        width = tf.cast(image_shape[1], tf.float32)

        crop_size = tf.cast(center_crop_size, tf.int32)
        crop_size = tf.where(
            crop_size > 0,
            tf.minimum(crop_size, tf.minimum(image_shape[0], image_shape[1])),
            tf.minimum(image_shape[0], image_shape[1]),
        )
        offset_y = (image_shape[0] - crop_size) // 2
        offset_x = (image_shape[1] - crop_size) // 2

        image = tf.image.crop_to_bounding_box(
            image,
            offset_y,
            offset_x,
            crop_size,
            crop_size,
        )
        image = tf.image.resize(
            image,
            (self.input_size, self.input_size),
            method=tf.image.ResizeMethod.BILINEAR,
        )
        image = tf.keras.applications.mobilenet_v2.preprocess_input(image)

        crop_size_f = tf.cast(crop_size, tf.float32)
        offset = tf.cast(tf.stack([offset_x, offset_y]), tf.float32)
        image_size = tf.stack([width, height])
        centroid_pixels = centroids * image_size
        cropped_centroids = (centroid_pixels - offset) / crop_size_f
        inside_crop = tf.reduce_all(
            (cropped_centroids >= 0.0) & (cropped_centroids < 1.0),
            axis=-1,
        )
        valid_centroids = centroid_mask & inside_crop

        grid_indices = tf.cast(
            tf.floor(cropped_centroids * tf.cast(self.grid_size, tf.float32)),
            tf.int32,
        )
        grid_indices = tf.clip_by_value(grid_indices, 0, self.grid_size - 1)
        grid_indices = tf.boolean_mask(grid_indices, valid_centroids)
        flat_indices = grid_indices[:, 1] * self.grid_size + grid_indices[:, 0]
        flat_indices = tf.unique(flat_indices).y
        grid_indices = tf.stack(
            [flat_indices // self.grid_size, flat_indices % self.grid_size],
            axis=-1,
        )

        object_grid = tf.scatter_nd(
            grid_indices,
            tf.ones((tf.shape(grid_indices)[0],), dtype=tf.float32),
            (self.grid_size, self.grid_size),
        )
        background_grid = 1.0 - object_grid
        target = tf.stack([background_grid, object_grid], axis=-1)
        return image, target

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


def center_crop_pil_image_and_centroids(img, centroids, crop_size):
    orig_w, orig_h = img.size
    crop_size = int(min(crop_size, orig_w, orig_h))
    left = (orig_w - crop_size) // 2
    top = (orig_h - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size

    cropped_centroids = []
    for x_norm, y_norm in centroids:
        x = x_norm * orig_w
        y = y_norm * orig_h
        if left <= x < right and top <= y < bottom:
            cropped_centroids.append(((x - left) / crop_size, (y - top) / crop_size))

    return img.crop((left, top, right, bottom)), cropped_centroids


def pad_centroids(centroids, max_objects=DEFAULT_MAX_OBJECTS):
    padded = np.zeros((max_objects, 2), dtype=np.float32)
    mask = np.zeros((max_objects,), dtype=bool)
    for index, centroid in enumerate(centroids[:max_objects]):
        padded[index] = centroid
        mask[index] = True
    return padded, mask


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


def iter_raw_hf_dataset(builder, hf_dataset, center_crop_size=None):
    for example in hf_dataset:
        yield builder.process_raw_example(
            example,
            center_crop_size=center_crop_size,
        )


def build_accelerated_tf_dataset_from_hf_dataset(
    builder,
    hf_dataset,
    center_crop_size=None,
    cache=None,
):
    output_signature = (
        tf.TensorSpec(shape=(None, None, 3), dtype=tf.uint8),
        tf.TensorSpec(shape=(builder.max_objects, 2), dtype=tf.float32),
        tf.TensorSpec(shape=(builder.max_objects,), dtype=tf.bool),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
    dataset = tf.data.Dataset.from_generator(
        lambda: iter_raw_hf_dataset(
            builder,
            hf_dataset,
            center_crop_size=center_crop_size,
        ),
        output_signature=output_signature,
    )
    dataset = dataset.map(
        builder.preprocess_raw_example,
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    if cache is not None:
        dataset = dataset.cache(cache)
    return dataset


def build_accelerated_tf_dataset_from_hf_datasets(
    builder,
    dataset_specs,
    center_crop_sizes=None,
    cache=None,
):
    center_crop_sizes = center_crop_sizes or {}
    datasets = []
    for repo_id, hf_dataset in dataset_specs:
        datasets.append(
            build_accelerated_tf_dataset_from_hf_dataset(
                builder,
                hf_dataset,
                center_crop_size=center_crop_sizes.get(repo_id),
            )
        )

    if not datasets:
        raise ValueError("At least one Hugging Face dataset is required")

    dataset = datasets[0]
    for next_dataset in datasets[1:]:
        dataset = dataset.concatenate(next_dataset)

    if cache is not None:
        dataset = dataset.cache(cache)
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
