from setuptools import setup, find_packages

setup(
    name="edgeai_nuvoton_core",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "ai-edge-litert",
        "datasets",
        "ethos-u-vela",
        "flatbuffers",
        "matplotlib",
        "numpy",
        "onnx",
        "onnx-graphsurgeon",
        "onnx-tf",
        "onnx2tf",
        "onnxruntime",
        "onnxsim",
        "pillow",
        "protobuf",
        "pyyaml",
        "sng4onnx",
        "tensorflow",
        "tf-keras",
    ],
)
