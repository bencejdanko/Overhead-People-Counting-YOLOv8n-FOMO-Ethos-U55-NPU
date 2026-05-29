from __future__ import annotations

import subprocess
from pathlib import Path


def convert_onnx_to_tflite_with_onnx2tf(
    onnx_path: str | Path,
    output_dir: str | Path,
    quantize_int8=False,
    representative_data_dir: str | Path | None = None,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "onnx2tf",
        "-i",
        str(onnx_path),
        "-o",
        str(output_dir),
    ]
    if quantize_int8:
        command.extend(["-oiqt"])
        if representative_data_dir is not None:
            command.extend(["-cind", str(representative_data_dir)])

    subprocess.run(command, check=True)
    tflite_files = sorted(output_dir.rglob("*.tflite"))
    if not tflite_files:
        raise FileNotFoundError(f"onnx2tf did not create a .tflite file in {output_dir}")
    return tflite_files[0]


def compile_tflite_with_vela(
    tflite_path: str | Path,
    output_dir: str | Path = "vela_output",
    vela_config: str | Path = "configs/default_vela.ini",
    accelerator_config="ethos-u55-256",
):
    command = [
        "vela",
        str(tflite_path),
        "--accelerator-config",
        accelerator_config,
        "--config",
        str(vela_config),
        "--output-dir",
        str(output_dir),
    ]
    subprocess.run(command, check=True)
    return Path(output_dir)
