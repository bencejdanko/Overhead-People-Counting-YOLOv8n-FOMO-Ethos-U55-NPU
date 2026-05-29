from __future__ import annotations

import shutil
import sys
from pathlib import Path


def replace_torch_activations_with_relu6(module):
    import torch.nn as nn

    for name, child in list(module.named_children()):
        if isinstance(child, (nn.SiLU, nn.ReLU, nn.LeakyReLU, nn.Hardswish)):
            setattr(module, name, nn.ReLU6(inplace=True))
        else:
            replace_torch_activations_with_relu6(child)
    return module


def train_libreyolo_model(
    data_yaml: str | Path,
    model_ref="LibreYOLOXn.pt",
    size="n",
    epochs=100,
    batch=16,
    imgsz=416,
    project="runs/yolo_train",
    name="libreyolo_nano",
    device="auto",
    relu6=False,
    **train_kwargs,
):
    repo_root = Path(__file__).resolve().parents[1]
    local_libreyolo = repo_root / "libreyolo"
    if local_libreyolo.exists():
        sys.path.insert(0, str(local_libreyolo))

    from libreyolo import LibreYOLO

    model = LibreYOLO(model_ref, size=size, device=device)
    if relu6:
        replace_torch_activations_with_relu6(model.model)

    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=project,
        name=name,
        device=device,
        **train_kwargs,
    )
    return model, results


def export_libreyolo_onnx(
    model,
    imgsz=416,
    simplify=True,
    opset=13,
    output_dir: str | Path | None = None,
):
    output_path = model.export(
        format="onnx",
        imgsz=imgsz,
        simplify=simplify,
        opset=opset,
    )
    output_path = Path(output_path)
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        destination = output_dir / output_path.name
        if output_path.resolve() != destination.resolve():
            shutil.copy2(output_path, destination)
        output_path = destination
    return output_path
