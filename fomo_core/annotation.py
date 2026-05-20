class CocoCentroidAnnotator:
    def __init__(self, category_id=0, category_name="person", merge_all_categories=False):
        self.category_id = category_id
        self.category_name = category_name
        self.merge_all_categories = merge_all_categories

    def get_true_centroids(self, pil_img, example_objects=None):
        orig_w, orig_h = pil_img.size
        if not example_objects:
            return []

        centroids = []
        for obj in self._iter_objects(example_objects):
            if not self._is_target_category(obj):
                continue

            bbox = obj.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x_min, y_min, width, height = bbox
            if width <= 0 or height <= 0:
                continue

            cx = x_min + (width / 2.0)
            cy = y_min + (height / 2.0)
            centroids.append((cx / orig_w, cy / orig_h))

        return centroids

    def _iter_objects(self, example_objects):
        if isinstance(example_objects, list):
            yield from example_objects
            return

        if isinstance(example_objects, dict):
            bboxes = example_objects.get("bbox", [])
            categories = (
                example_objects.get("category_id")
                or example_objects.get("category")
                or [None] * len(bboxes)
            )
            category_names = example_objects.get("category", [None] * len(bboxes))
            source_categories = example_objects.get("source_category", [None] * len(bboxes))
            for bbox, category, category_name, source_category in zip(
                bboxes,
                categories,
                category_names,
                source_categories,
            ):
                yield {
                    "bbox": bbox,
                    "category_id": category,
                    "category": category_name,
                    "source_category": source_category,
                }

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


COCOPersonAnnotator = CocoCentroidAnnotator
