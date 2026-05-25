from collections.abc import Sequence


class CocoCentroidAnnotator:
    def __init__(
        self,
        category_id=0,
        category_name="person",
        merge_all_categories=False,
        centroid_box_fields=("bbox",),
    ):
        self.category_id = category_id
        self.category_name = category_name
        self.merge_all_categories = merge_all_categories
        self.centroid_box_fields = centroid_box_fields

    def get_true_centroids(self, pil_img, example_objects=None):
        orig_w, orig_h = pil_img.size
        if not example_objects:
            return []

        centroids = []
        for obj in self._iter_objects(example_objects):
            if not self._is_target_category(obj):
                continue

            centroid = self._get_object_centroid(obj)
            if centroid is None:
                continue
            cx, cy = centroid
            centroids.append((cx / orig_w, cy / orig_h))

        return centroids

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
            category_names = example_objects.get("category", [None] * object_count)
            source_categories = example_objects.get("source_category", [None] * object_count)
            for index in range(object_count):
                obj = {
                    "category_id": self._list_get(categories, index),
                    "category": self._list_get(category_names, index),
                    "source_category": self._list_get(source_categories, index),
                }
                for key, values in example_objects.items():
                    if self._is_sequence(values):
                        obj[key] = self._list_get(values, index)
                yield obj

    def _get_object_centroid(self, obj):
        for field in self.centroid_box_fields:
            box = obj.get(field)
            centroid = self._centroid_from_box(box, box_format=field)
            if centroid is not None:
                return centroid
        return None

    def _centroid_from_box(self, box, box_format="bbox"):
        if box is None:
            return None

        if isinstance(box, dict):
            return self._centroid_from_box_dict(box)

        values = self._flatten_numeric_values(box)
        if not values:
            return None

        if box_format == "bbox" and len(values) == 4:
            x_min, y_min, width, height = values
            if width <= 0 or height <= 0:
                return None
            return x_min + (width / 2.0), y_min + (height / 2.0)

        if len(values) == 5:
            return values[0], values[1]

        if len(values) >= 6 and len(values) % 2 == 0:
            xs = values[0::2]
            ys = values[1::2]
            return sum(xs) / len(xs), sum(ys) / len(ys)

        if len(values) >= 2:
            return values[0], values[1]

        return None

    def _centroid_from_box_dict(self, box):
        for x_key, y_key in (
            ("cx", "cy"),
            ("center_x", "center_y"),
            ("x_center", "y_center"),
            ("x", "y"),
        ):
            if x_key in box and y_key in box:
                return float(box[x_key]), float(box[y_key])

        for key in ("points", "vertices", "polygon", "bbox"):
            if key in box:
                return self._centroid_from_box(box[key], box_format=key)

        return None

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

    def _is_target_category(self, obj):
        if self.merge_all_categories:
            return True

        category_id = obj.get("category_id")
        if category_id == self.category_id:
            return True

        for key in ("category", "source_category"):
            category = obj.get(key)
            if isinstance(category, str) and category.lower() == self.category_name.lower():
                return True

        return False


class RadialBoxCentroidAnnotator(CocoCentroidAnnotator):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("centroid_box_fields", ("ritbox", "rotated_box", "bbox"))
        super().__init__(*args, **kwargs)


COCOPersonAnnotator = CocoCentroidAnnotator
