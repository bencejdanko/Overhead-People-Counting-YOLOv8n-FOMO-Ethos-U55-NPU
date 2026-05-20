from setuptools import setup, find_packages

setup(
    name="fomo_core",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "transformers",
        "accelerate",
        "opencv-python",
        "matplotlib",
        "datasets"
    ],
)