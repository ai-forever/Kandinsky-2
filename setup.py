from setuptools import setup

setup(
    name="kandinsky2",
    packages=[
        "kandinsky2",
        "kandinsky2/vqgan",
        "kandinsky2/model"
    ],
    install_requires=[
        "Pillow",
        "attrs",
        "torch",
        "filelock",
        "requests",
        "tqdm",
        "ftfy",
        "regex",
        "numpy",
        "blobfile",
        "transformers==4.23.1",
        "torchvision",
        "omegaconf",
        "pytorch_lightning",
        "einops",
        "sentencepiece",
  
    ],
    author="",
)
