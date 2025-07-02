# Benchmark Framework cho Phân Loại Ảnh

Framework này cung cấp các công cụ để đánh giá (benchmark) nhiều mô hình phân loại ảnh trên các tập dữ liệu khác nhau. Nó hỗ trợ cả mô hình TensorFlow/Keras và PyTorch, cho phép so sánh dễ dàng giữa các kiến trúc và framework khác nhau.

## Tính Năng

- Liệt kê các tập dữ liệu và mô hình có sẵn
- Tải xuống và giải nén tập dữ liệu từ máy chủ từ xa
- Huấn luyện và đánh giá nhiều mô hình trên tập dữ liệu được chỉ định
- Vẽ biểu đồ lịch sử huấn luyện để so sánh trực quan
- Thu thập và hiển thị các chỉ số hiệu suất dưới dạng bảng

## Yêu Cầu

Để sử dụng framework này, hãy đảm bảo bạn đã cài đặt các thư viện Python sau:

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

Bạn có thể cài đặt các gói cần thiết bằng pip:

```bash
pip install tensorflow keras torch torchvision huggingface_hub timm scikit-learn pandas numpy matplotlib
```

## Bắt Đầu Nhanh

1. **Nhập framework:**

```python
import framework
```

2. **Liệt kê các tài nguyên có sẵn:**

```python
resources = framework.list_available_resources()
print(resources)
```

Lệnh này sẽ hiển thị danh sách các tập dữ liệu và mô hình có thể sử dụng với framework.

3. **Tải xuống và giải nén tập dữ liệu:**

```python
paths = framework.download_and_extract_zips(['dataset_name'], download_dir='datasets', extract_dir='data')
```

Thay `'dataset_name'` bằng tên của tập dữ liệu bạn muốn sử dụng. Tập dữ liệu sẽ được tải xuống và giải nén vào các thư mục được chỉ định.

4. **Chọn các mô hình để đánh giá:**

```python
models = ['model1', 'model2', ...]
```

Chọn các mô hình bạn muốn huấn luyện và đánh giá từ danh sách các mô hình được hỗ trợ.

5. **Chạy benchmark:**

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

Hàm này sẽ huấn luyện từng mô hình trên tập dữ liệu, đánh giá hiệu suất của chúng và trả về một DataFrame chứa kết quả.

6. **Xem kết quả:**

```python
print(metrics_df)
```

DataFrame sẽ chứa các chỉ số hiệu suất như precision, recall, F1-score và accuracy cho từng mô hình.

## Các Hàm Chính

### `list_available_resources()`

Trả về một từ điển chứa các tập dữ liệu và mô hình có sẵn để sử dụng với framework.

### `download_and_extract_zips(selected_datasets, download_dir='datasets', extract_dir='data')`

Tải xuống và giải nén các tập dữ liệu được chỉ định từ máy chủ từ xa.

- **selected_datasets**: Danh sách tên các tập dữ liệu cần tải.
- **download_dir**: Thư mục để lưu các file zip.
- **extract_dir**: Thư mục để giải nén tập dữ liệu.

Trả về danh sách các đường dẫn đến các tập dữ liệu đã được giải nén.

### `benchmark_model(...)`

Huấn luyện và đánh giá nhiều mô hình trên một tập dữ liệu đã cho.

**Tham số:**

- `models_to_train`: Danh sách tên các mô hình cần huấn luyện.
- `dataset_path`: Đường dẫn đến thư mục tập dữ liệu.
- `target_size`: Tuple kích thước ảnh cần resize (chiều cao, chiều rộng).
- `color_mode`: 'rgb' hoặc 'grayscale'.
- `class_mode`: 'categorical', 'binary', v.v.
- `batch_size`: Kích thước batch cho huấn luyện.
- `shuffle`: Có xáo trộn dữ liệu hay không.
- `seed`: Hạt giống ngẫu nhiên để tái tạo kết quả.
- `validation_split`: Tỷ lệ dữ liệu dùng cho validation.
- `test_split`: Tỷ lệ dữ liệu dùng cho kiểm tra.
- `num_workers`: Số lượng worker để tải dữ liệu.
- `input_shape`: Hình dạng đầu vào cho các mô hình.
- `epochs`: Số epoch tối đa để huấn luyện.
- `patience`: Độ kiên nhẫn cho early stopping.
- `learning_rate`: Tốc độ học của optimizer.
- `fine_tune`: Có tinh chỉnh các lớp pretrained hay không.
- `checkpoint_path`: Đường dẫn để lưu checkpoint của mô hình.

Trả về một DataFrame pandas chứa các chỉ số hiệu suất cho từng mô hình.

## Các Mô Hình Được Hỗ Trợ

Framework hỗ trợ các mô hình sau:

- **EfficientNetB0**
- **Xception**
- **ResNet50**
- **ResNet101**
- **ResNet152**
- **DenseNet121**
- **VGG16**
- **ConvNeXtBase**
- **CNN** (mô hình CNN tùy chỉnh)
- **ViT_B16** (Vision Transformer)
- **EfficientNetV2L**
- **CoAtNet**

Các mô hình như **ViT_B16** và **CoAtNet** được triển khai bằng PyTorch và TIMM, trong khi các mô hình khác dựa trên TensorFlow/Keras.

## Giải Thích Kết Quả

Hàm `benchmark_model` trả về một DataFrame pandas với các cột sau:

- **Mô hình**: Tên mô hình
- **Số vòng lặp thực tế**: Số epoch thực tế đã huấn luyện (có thể ít hơn `epochs` do early stopping)
- **Patience**: Giá trị patience dùng cho early stopping
- **Precision (%)**: Độ chính xác trung bình có trọng số trên tập kiểm tra
- **Recall (%)**: Độ nhạy trung bình có trọng số trên tập kiểm tra
- **F1-score (%)**: Điểm F1 trung bình có trọng số trên tập kiểm tra
- **Accuracy (%)**: Độ chính xác trên tập kiểm tra
- **Thời gian/epoch (s)**: Thời gian trung bình mỗi epoch tính bằng giây

Các chỉ số này cho phép so sánh toàn diện về hiệu suất và hiệu quả huấn luyện của các mô hình khác nhau.

## Biểu Đồ Lịch Sử Huấn Luyện

Sau khi chạy benchmark, framework tự động tạo các biểu đồ so sánh lịch sử huấn luyện của tất cả các mô hình. Các biểu đồ này bao gồm:

- **Loss qua Epoch**: Loss huấn luyện và validation cho từng mô hình.
- **Accuracy qua Epoch**: Accuracy huấn luyện và validation cho từng mô hình.

Những biểu đồ này giúp hiểu cách mỗi mô hình học theo thời gian và so sánh hành vi hội tụ của chúng.

## Ví Dụ

Để xem ví dụ đầy đủ, hãy tham khảo notebook `Run_framework.ipynb`, trong đó trình bày cách sử dụng framework từng bước.

## Ghi Chú

- Đảm bảo hệ thống của bạn có đủ tài nguyên (CPU/GPU, bộ nhớ) để xử lý việc huấn luyện, đặc biệt với các mô hình và tập dữ liệu lớn.
- Framework sử dụng early stopping dựa trên validation loss để tránh overfitting và tiết kiệm thời gian tính toán.
- Đối với các mô hình PyTorch, framework tự động chọn thiết bị phù hợp (CPU hoặc GPU).
- Cả feature extraction và fine-tuning đều được hỗ trợ cho các mô hình pretrained bằng cách cài đặt tham số `fine_tune`.