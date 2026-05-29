from setuptools import setup, find_packages

setup(
    name="edgeai_nuvoton_core",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "tensorflow",
        "matplotlib",
        "datasets",
        "pillow",
        "onnx",
        "onnxsim",
        "onnxruntime",
        "onnx2tf",
        "tensorflow",
        "onnx-tf",
        "pyyaml",
        "ethos-u-vela"
    ],
)
