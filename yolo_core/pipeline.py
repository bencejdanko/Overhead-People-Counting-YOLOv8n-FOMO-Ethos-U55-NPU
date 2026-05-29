from __future__ import annotations

import shutil
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import yaml
from PIL import Image


DEFAULT_IMAGE_KEY = "image"
DEFAULT_OBJECTS_KEY = "objects"
DEFAULT_BOX_FIELDS = ("bbox", "box", "xywh", "ritbox", "rotated_box")


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class YoloDetectionAnnotator:
    """Extract per-class YOLO boxes from Hugging Face object annotations."""

    def __init__(
        self,
        class_names: Sequence[str] | None = None,
        category_id_to_class_id: Mapping[int, int] | None = None,
        merge_all_categories: bool = False,
        default_class_name: str = "object",
        box_fields: Sequence[str] = DEFAULT_BOX_FIELDS,
    ):
        self.class_names = list(class_names or [])
        self.category_id_to_class_id = dict(category_id_to_class_id or {})
        self.merge_all_categories = merge_all_categories
        self.default_class_name = default_class_name
        self.box_fields = tuple(box_fields)

    def get_boxes(self, pil_img, example_objects=None) -> list[YoloBox]:
        if not example_objects:
            return []

        width, height = pil_img.size
        boxes = []
        for obj in self._iter_objects(example_objects):
            class_id = self._class_id_for_object(obj)
            if class_id is None:
                continue

            xywh = self._get_xywh(obj)
            if xywh is None:
                continue

            x, y, w, h = xywh
            x_min = max(0.0, x)
            y_min = max(0.0, y)
            x_max = min(float(width), x + w)
            y_max = min(float(height), y + h)
            clipped_w = x_max - x_min
            clipped_h = y_max - y_min
            if clipped_w <= 0 or clipped_h <= 0:
                continue

            boxes.append(
                YoloBox(
                    class_id=class_id,
                    x_center=((x_min + x_max) / 2.0) / width,
                    y_center=((y_min + y_max) / 2.0) / height,
                    width=clipped_w / width,
                    height=clipped_h / height,
                )
            )
        return boxes

    def names(self) -> list[str]:
        if self.class_names:
            return self.class_names
        return [self.default_class_name]

    def _class_id_for_object(self, obj):
        if self.merge_all_categories:
            return 0

        category_id = obj.get("category_id")
        if category_id is not None:
            try:
                category_id = int(category_id)
            except (TypeError, ValueError):
                category_id = None
        if category_id in self.category_id_to_class_id:
            return self.category_id_to_class_id[category_id]

        category = obj.get("category") or obj.get("source_category")
        if isinstance(category, str) and self.class_names:
            normalized = category.lower()
            for index, name in enumerate(self.class_names):
                if normalized == name.lower():
                    return index

        if category_id is not None and not self.class_names:
            return category_id

        return None

    def _get_xywh(self, obj):
        for field in self.box_fields:
            box = obj.get(field)
            xywh = self._xywh_from_box(box, box_format=field)
            if xywh is not None:
                return xywh
        return None

    def _xywh_from_box(self, box, box_format="bbox"):
        if box is None:
            return None

        if isinstance(box, dict):
            if all(key in box for key in ("x", "y", "width", "height")):
                return (
                    float(box["x"]),
                    float(box["y"]),
                    float(box["width"]),
                    float(box["height"]),
                )
            if all(key in box for key in ("x_min", "y_min", "x_max", "y_max")):
                x_min = float(box["x_min"])
                y_min = float(box["y_min"])
                x_max = float(box["x_max"])
                y_max = float(box["y_max"])
                return x_min, y_min, x_max - x_min, y_max - y_min
            for key in ("bbox", "box", "xywh"):
                if key in box:
                    return self._xywh_from_box(box[key], box_format=key)

        values = self._flatten_numeric_values(box)
        if len(values) == 4:
            x, y, w, h = values
            if box_format in {"xyxy", "corners"}:
                return x, y, w - x, h - y
            return x, y, w, h

        if len(values) >= 6 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]
            x_min = min(xs)
            y_min = min(ys)
            return x_min, y_min, max(xs) - x_min, max(ys) - y_min

        return None

    def _iter_objects(self, example_objects):
        if isinstance(example_objects, list):
            yield from example_objects
            return

        if isinstance(example_objects, dict):
            object_count = max(
                (
                    len(value)
                    for value in example_objects.values()
                    if self._is_sequence(value)
                ),
                default=0,
            )
            categories = (
                example_objects.get("category_id")
                or example_objects.get("category")
                or [None] * object_count
            )
            for index in range(object_count):
                obj = {
                    "category_id": self._list_get(categories, index),
                    "category": self._list_get(
                        example_objects.get("category", []), index
                    ),
                    "source_category": self._list_get(
                        example_objects.get("source_category", []), index
                    ),
                }
                for key, values in example_objects.items():
                    if self._is_sequence(values):
                        obj[key] = self._list_get(values, index)
                yield obj

    def _flatten_numeric_values(self, value):
        if isinstance(value, (int, float)):
            return [float(value)]
        if self._is_sequence(value):
            values = []
            for item in value:
                values.extend(self._flatten_numeric_values(item))
            return values
        return []

    def _list_get(self, values, index):
        if self._is_sequence(values) and index < len(values):
            return values[index]
        return None

    def _is_sequence(self, value):
        return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


class YoloDatasetExporter:
    def __init__(
        self,
        annotator: YoloDetectionAnnotator,
        image_key=DEFAULT_IMAGE_KEY,
        objects_key=DEFAULT_OBJECTS_KEY,
        image_format="jpg",
        center_crop_size=None,
    ):
        self.annotator = annotator
        self.image_key = image_key
        self.objects_key = objects_key
        self.image_format = image_format.lower().lstrip(".")
        self.center_crop_size = center_crop_size

    def export_split(
        self,
        hf_dataset,
        output_dir: str | Path,
        split_name: str,
        filename_prefix="",
    ):
        split_dir = Path(output_dir) / split_name
        images_dir = split_dir / "images"
        labels_dir = split_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        image_count = 0
        box_count = 0
        for index, example in enumerate(hf_dataset):
            image = load_pil_image(example[self.image_key])
            boxes = self.annotator.get_boxes(
                image,
                example_objects=example.get(self.objects_key),
            )

            if self.center_crop_size is not None:
                image, boxes = center_crop_image_and_boxes(
                    image,
                    boxes,
                    self.center_crop_size,
                )

            stem = f"{filename_prefix}{index:08d}"
            image.save(images_dir / f"{stem}.{self.image_format}")
            (labels_dir / f"{stem}.txt").write_text(
                "".join(
                    f"{box.class_id} {box.x_center:.8f} {box.y_center:.8f} "
                    f"{box.width:.8f} {box.height:.8f}\n"
                    for box in boxes
                ),
                encoding="utf-8",
            )
            image_count += 1
            box_count += len(boxes)

        return {"images": image_count, "boxes": box_count}

    def write_data_yaml(self, output_dir: str | Path, splits: Sequence[str]):
        output_dir = Path(output_dir)
        data = {
            "path": str(output_dir.resolve()),
            "train": "train/images",
            "val": "validation/images" if "validation" in splits else "val/images",
            "test": "test/images" if "test" in splits else None,
            "nc": len(self.annotator.names()),
            "names": self.annotator.names(),
        }
        data = {key: value for key, value in data.items() if value is not None}
        data_yaml = output_dir / "data.yaml"
        data_yaml.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
        return data_yaml


def export_hf_datasets_to_yolo(
    dataset_splits: Mapping[str, Iterable],
    output_dir: str | Path,
    annotator: YoloDetectionAnnotator,
    center_crop_size=None,
    overwrite=False,
):
    output_dir = Path(output_dir)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    exporter = YoloDatasetExporter(
        annotator=annotator,
        center_crop_size=center_crop_size,
    )
    stats = {}
    for split_name, dataset in dataset_splits.items():
        stats[split_name] = exporter.export_split(dataset, output_dir, split_name)
    data_yaml = exporter.write_data_yaml(output_dir, tuple(dataset_splits.keys()))
    return data_yaml, stats


def export_hf_dataset_specs_to_yolo(
    dataset_split_specs: Mapping[str, Sequence[tuple[str, Iterable]]],
    output_dir: str | Path,
    annotator: YoloDetectionAnnotator,
    center_crop_sizes: Mapping[str, int] | None = None,
    overwrite=False,
):
    output_dir = Path(output_dir)
    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    center_crop_sizes = center_crop_sizes or {}
    stats = {}
    for split_name, dataset_specs in dataset_split_specs.items():
        stats[split_name] = {"images": 0, "boxes": 0}
        for repo_id, dataset in dataset_specs:
            exporter = YoloDatasetExporter(
                annotator=annotator,
                center_crop_size=center_crop_sizes.get(repo_id),
            )
            prefix = _safe_filename_prefix(repo_id)
            split_stats = exporter.export_split(
                dataset,
                output_dir,
                split_name,
                filename_prefix=prefix,
            )
            stats[split_name]["images"] += split_stats["images"]
            stats[split_name]["boxes"] += split_stats["boxes"]

    exporter = YoloDatasetExporter(annotator=annotator)
    data_yaml = exporter.write_data_yaml(output_dir, tuple(dataset_split_specs.keys()))
    return data_yaml, stats


def _safe_filename_prefix(value):
    safe = "".join(char if char.isalnum() else "_" for char in value)
    return f"{safe}_"


def load_pil_image(image_value):
    if isinstance(image_value, Image.Image):
        return image_value.convert("RGB")

    if isinstance(image_value, dict):
        if image_value.get("bytes") is not None:
            return Image.open(BytesIO(image_value["bytes"])).convert("RGB")
        if image_value.get("path"):
            return Image.open(image_value["path"]).convert("RGB")

    raise TypeError(f"Unsupported image value: {type(image_value)!r}")


def center_crop_image_and_boxes(image, boxes, crop_size):
    orig_w, orig_h = image.size
    crop_size = int(min(crop_size, orig_w, orig_h))
    left = (orig_w - crop_size) // 2
    top = (orig_h - crop_size) // 2
    cropped = image.crop((left, top, left + crop_size, top + crop_size))

    kept = []
    for box in boxes:
        cx = box.x_center * orig_w
        cy = box.y_center * orig_h
        if not (left <= cx < left + crop_size and top <= cy < top + crop_size):
            continue
        kept.append(
            YoloBox(
                class_id=box.class_id,
                x_center=(cx - left) / crop_size,
                y_center=(cy - top) / crop_size,
                width=min(1.0, box.width * orig_w / crop_size),
                height=min(1.0, box.height * orig_h / crop_size),
            )
        )

    return cropped, kept
