from setuptools import setup

setup(
    name="natalle",
    packages=[
        "natalle",
        "natalle.vqgan"
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
