from setuptools import setup

setup(
    name="natalle",
    packages=[
        "natalle",
        "natalle/vqgan",
	"natalle/model"
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
	"transformers",
	"torchvision",
	"omegaconf"
	    
    ],
    author="",
)
