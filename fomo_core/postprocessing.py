import numpy as np


def extract_fomo_peaks(prediction, threshold=0.5, min_distance=2, object_class=1):
    if prediction.ndim == 3:
        heatmap = prediction[:, :, object_class]
    elif prediction.ndim == 2:
        heatmap = prediction
    else:
        raise ValueError(f"Expected a 2D heatmap or 3D class map, got shape {prediction.shape}")

    peaks = []
    candidates = np.argwhere(heatmap >= threshold)
    candidates = sorted(
        candidates,
        key=lambda cell: heatmap[cell[0], cell[1]],
        reverse=True,
    )

    for grid_y, grid_x in candidates:
        too_close = False
        for peak in peaks:
            dy = grid_y - peak["grid_y"]
            dx = grid_x - peak["grid_x"]
            if (dx * dx + dy * dy) ** 0.5 < min_distance:
                too_close = True
                break

        if too_close:
            continue

        peaks.append(
            {
                "grid_x": int(grid_x),
                "grid_y": int(grid_y),
                "score": float(heatmap[grid_y, grid_x]),
            }
        )

    return peaks


def peaks_to_image_points(peaks, input_size=192, grid_size=24):
    points = []
    for peak in peaks:
        x = (peak["grid_x"] + 0.5) * input_size / grid_size
        y = (peak["grid_y"] + 0.5) * input_size / grid_size
        points.append({**peak, "x": x, "y": y})
    return points
