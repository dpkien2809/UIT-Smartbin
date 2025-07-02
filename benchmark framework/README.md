# 📊 Benchmark Framework for Image Classification

This framework provides tools to benchmark multiple image classification models on various datasets. It supports both TensorFlow/Keras and PyTorch models, allowing for easy comparison across different architectures and frameworks.

## ⭐ Features

- List available datasets and models
- Download and extract datasets from a remote server
- Train and evaluate multiple models on a specified dataset
- Plot training history for visual comparison
- Collect and display performance metrics in a tabular format

## 🛠️ Requirements

To use this framework, ensure you have the following Python libraries installed:

- Python 3.6+
- TensorFlow 2.x
- Keras
- PyTorch
- Torchvision
- Hugging Face Hub
- Timm
- Scikit-learn
- Pandas
- NumPy
- Matplotlib

You can install the required packages using pip:

```bash
pip install tensorflow keras torch torchvision huggingface_hub timm scikit-learn pandas numpy matplotlib
```

## 🚀 Quick Start

1. **Import the framework:**

```python
import framework
```

2. **List available resources:**

```python
resources = framework.list_available_resources()
print(resources)
```

This will display the datasets and models available for use with the framework.

3. **Download and extract datasets:**

```python
paths = framework.download_and_extract_zips(['dataset_name'], download_dir='datasets', extract_dir='data')
```

Replace `'dataset_name'` with the name of the dataset you wish to use. The dataset will be downloaded and extracted to the specified directories.

4. **Select models to benchmark:**

```python
models = ['model1', 'model2', ...]
```

Choose the models you want to train and evaluate from the list of supported models.

5. **Run the benchmark:**

```python
metrics_df = framework.benchmark_model(
    models_to_train=models,
    dataset_path=paths[0],
    target_size=(224, 224),
    batch_size=32,
    epochs=10,
    patience=5,
    learning_rate=1e-3,
    fine_tune=False
)
```

This function will train each model on the dataset, evaluate their performance, and return a DataFrame with the results.

6. **View the results:**

```python
print(metrics_df)
```

The DataFrame will contain performance metrics such as precision, recall, F1-score, and accuracy for each model.

## ⚙️ Main Functions

### `list_available_resources()`

Returns a dictionary containing the available datasets and models for use with the framework.

### `download_and_extract_zips(selected_datasets, download_dir='datasets', extract_dir='data')`

Downloads and extracts the specified datasets from a remote server.

- **selected_datasets**: List of dataset names to download.
- **download_dir**: Directory to save the zip files.
- **extract_dir**: Directory to extract the datasets.

Returns a list of paths to the extracted datasets.

### `benchmark_model(...)`

Trains and evaluates multiple models on a given dataset.

**Parameters:**

- `models_to_train`: List of model names to train.
- `dataset_path`: Path to the dataset directory.
- `target_size`: Tuple of image resize dimensions (height, width).
- `color_mode`: 'rgb' or 'grayscale'.
- `class_mode`: 'categorical', 'binary', etc.
- `batch_size`: Batch size for training.
- `shuffle`: Whether to shuffle the data.
- `seed`: Random seed for reproducibility.
- `validation_split`: Ratio of data to use for validation.
- `test_split`: Ratio of data to use for testing.
- `num_workers`: Number of workers for data loading.
- `input_shape`: Input shape for the models.
- `epochs`: Maximum number of training epochs.
- `patience`: Patience for early stopping.
- `learning_rate`: Learning rate for the optimizer.
- `fine_tune`: Whether to fine-tune pretrained layers.
- `checkpoint_path`: Path to save model checkpoints.

Returns a pandas DataFrame containing performance metrics for each model.

## 🧠 Supported Models

The framework supports the following models:

- **EfficientNetB0**
- **Xception**
- **ResNet50**
- **ResNet101**
- **ResNet152**
- **DenseNet121**
- **VGG16**
- **ConvNeXtBase**
- **CNN** (custom CNN model)
- **ViT_B16** (Vision Transformer)
- **EfficientNetV2L**
- **CoAtNet**

Models like **ViT_B16** and **CoAtNet** are implemented using PyTorch and TIMM, while others are based on TensorFlow/Keras.

## 📊 Result Explanation

The `benchmark_model` function returns a pandas DataFrame with the following columns:

- **Model**: Name of the model
- **Actual Epochs**: Number of epochs actually trained (may be less than `epochs` due to early stopping)
- **Patience**: Patience value used for early stopping
- **Precision (%)**: Weighted average precision on the test set
- **Recall (%)**: Weighted average recall on the test set
- **F1-score (%)**: Weighted average F1-score on the test set
- **Accuracy (%)**: Accuracy on the test set
- **Time per Epoch (s)**: Average time per epoch in seconds

These metrics provide a comprehensive comparison of the performance and training efficiency of different models.

## 📈 Training History Plots

After running the benchmark, the framework automatically generates plots comparing the training history of all models. These plots include:

- **Loss over Epochs**: Training and validation loss for each model.
- **Accuracy over Epochs**: Training and validation accuracy for each model.

These visualizations help understand how each model learns over time and compare their convergence behavior.

## 📝 Example

For a complete example, refer to the `Run_framework.ipynb` notebook, which demonstrates how to use the framework step-by-step.

## 📌 Notes

- Ensure your system has sufficient resources (CPU/GPU, memory) to handle the training, especially with large models and datasets.
- The framework uses early stopping based on validation loss to prevent overfitting and save computation time.
- For PyTorch models, the framework automatically selects the appropriate device (CPU or GPU).
- Both feature extraction and fine-tuning are supported for pretrained models by setting the `fine_tune` parameter.
