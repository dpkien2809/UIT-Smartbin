# Dataloader_for_Smartbin
<img src="https://raw.githubusercontent.com/howtodie123/howtodie123/readme.io/image/Garbage_image.jpg" alt="Python">


# Introduction
Dataloader for Garbage is a library designed to streamline the process of downloading and loading data for researchers. It aims to provide quick and efficient access to datasets related to garbage, optimizing researchers' workflow during data exploration and analysis."

# Installation
```sh
pip install git+https://github.com/howtodie123/Dataloader_for_Smartbin.git
```


# Instructions

- Get list of dataset names:
```python
from Dataloader import Dataset
list_data = Dataset.list_of_dataset()
```

# Download dataset
Use array names of dataset to download dataset in 1 times 

```python
selected_datasets = [              # Array for download dataset
    'garbage_classification_3',
    'Garbage_classification_2'
]
download_dir = '/content/Dataset' # folder for download dataset file zip
extract_dir = '/content/data' # folder for extract dataset
Download = Dataset.download_and_extract_zips(selected_datasets, download_dir,extract_dir)  
```

- Choose any dataset from the list of path dataset. In the code above, it will print out for you the paths of the folders containing the datasets you downloaded. Use that path to load dataset and split train_images , val_images,test_image. 
```python
train_images, val_images, test_images = Dataset.load_dataset('/content/data/Garbage_classification_2')
```
# Get Data visualization using matplotlib 
- Can get visualization for data after load dataset, you can put train/val/test data after load dataset into this function to see plot.
```python
Dataset.Data_visualization(train_images)
```
# Data infomation
- If you want to know more about your data to train, valid , test , you can use this function to see many variable infomation.
```python
Dataset.Data_info(train_images)
```
