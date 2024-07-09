from setuptools import setup, find_packages

setup(
    name="dataloader",
    version="1.0",
    packages=find_packages(),
    install_requires=[
        "pandas",  
    ],
    author="Nguyen Le Tien Phat",
    author_email="",
    description="Dataloader for Garbage is a library designed to streamline the process of downloading and loading data for researchers. It aims to provide quick and efficient access to datasets related to garbage, optimizing researchers' workflow during data exploration and analysis.",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
