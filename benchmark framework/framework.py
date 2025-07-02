# System and utilities
import os
import time
import copy
import random
import datetime
import zipfile
from pathlib import Path
from collections import defaultdict

# Data and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import (
    Input, Dense, Dropout, GlobalAveragePooling2D,
    RandomFlip, RandomRotation, RandomZoom,
    Conv2D, BatchNormalization, MaxPooling2D, Flatten, Rescaling
)
from tensorflow.keras.applications import (
    EfficientNetB0, Xception, ResNet50, ResNet101, ResNet152, EfficientNetV2L,
    DenseNet121, VGG16, ConvNeXtBase
)
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input as efficientnet_v2_preprocess
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess
from tensorflow.keras.applications.resnet import preprocess_input as resnet_preprocess
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg16_preprocess
from tensorflow.keras.applications.xception import preprocess_input as xception_preprocess
from tensorflow.keras.applications.convnext import preprocess_input as convnext_preprocess

# Keras CV Attention Models
import keras_cv_attention_models
from keras_cv_attention_models import coatnet

# Sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support,
    accuracy_score
)

# PyTorch and related libraries
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms

# Hugging Face
from huggingface_hub import hf_hub_download

# Timm
import timm



class CustomImageDataset(Dataset):
    """
    Custom Dataset class để load từ DataFrame
    """
    def __len__(self):
        return len(self.dataframe)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.dataframe.iloc[idx]['filepath']
        image = Image.open(img_path).convert('RGB')
        
        # Load label
        label_str = self.dataframe.iloc[idx]['label']
        label = self.label_encoder.transform([label_str])[0]
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        return image, label
    def __init__(self, dataframe, transform=None, label_encoder=None):
        """
        Args:
            dataframe: DataFrame storage 'filepath' và 'label'
            transform: torchvision transforms
            label_encoder: sklearn LabelEncoder to encode labels
        """
        self.dataframe = dataframe.reset_index(drop=True)
        self.transform = transform
        self.label_encoder = label_encoder
        
        # Nếu chưa có label encoder, tạo mới
        if self.label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.label_encoder.fit(self.dataframe['label'])

def setup_timm_model(model_name, num_classes, device, freeze_backbone=True):
    """
    Setup TIMM model fine-tuning configuration
    """
    
    model = timm.create_model(model_name, pretrained=False)
    model = model.to(device)
    
    # Configure classifier head based on model type
    if 'vit' in model_name.lower():
        # Vision Transformer
        saved_path = hf_hub_download('timm/vit_base_patch16_224.augreg2_in21k_ft_in1k', 'pytorch_model.bin', library_name='timm')
        state_dict = torch.load(saved_path)
        model.load_state_dict(state_dict)
        if hasattr(model, 'head'):
            in_features = model.head.in_features
            model.head = nn.Linear(in_features, num_classes).to(device)
        else:
            model.head = nn.Linear(768, num_classes).to(device)
            
    elif 'coatnet' in model_name.lower():
        # CoAtNet
        saved_path = hf_hub_download('timm/coatnet_2_rw_224.sw_in12k', 'pytorch_model.bin', library_name='timm')
        state_dict = torch.load(saved_path)
        model.load_state_dict(state_dict)
        if hasattr(model, 'head') and hasattr(model.head, 'fc'):
            in_features = model.head.fc.in_features
            model.head.fc = nn.Linear(in_features, num_classes).to(device)
        else:
            # Try different head configurations
            if hasattr(model, 'classifier'):
                in_features = model.classifier.in_features
                model.classifier = nn.Linear(in_features, num_classes).to(device)
            else:
                model.head = nn.Linear(1024, num_classes).to(device)  # Common CoAtNet feature size
    else:
        # Generic approach
        if hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_classes).to(device)
        elif hasattr(model, 'head'):
            if hasattr(model.head, 'in_features'):
                in_features = model.head.in_features
                model.head = nn.Linear(in_features, num_classes).to(device)
            else:
                model.head = nn.Linear(1000, num_classes).to(device)
    
    # Freeze backbone if requested
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze head/classifier layers
        head_unfrozen = False
        
        # Check for head with nested structure (like CoAtNet)
        if hasattr(model, 'head'):
            if hasattr(model.head, 'fc'):
                # CoAtNet case: head.fc
                for param in model.head.fc.parameters():
                    param.requires_grad = True
                head_unfrozen = True

            elif hasattr(model.head, 'parameters'):
                # Standard head case
                for param in model.head.parameters():
                    param.requires_grad = True
                head_unfrozen = True
 
        
        # Check for classifier
        if hasattr(model, 'classifier'):
            for param in model.classifier.parameters():
                param.requires_grad = True
            head_unfrozen = True

        
        if not head_unfrozen:
            print("⚠️ Warning: No head/classifier found to unfreeze")
            
    else:
        print("🔓 All parameters are trainable")
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"📊 Trainable parameters: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return model

def train_model_optimized(model, criterion, optimizer, dataloaders, device, num_epochs=25, patience=10):
    """
    Optimized training function với đầy đủ features
    """
    # Initialize tracking variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    epochs_no_improve = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    # History để lưu metrics
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'train_accuracy': [], 
        'val_accuracy': []
    }
        
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        
        print(f'\nEpoch {epoch + 1}/{num_epochs}')
        print('-' * 40)
        
        # Each epoch has training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0
            
            # Iterate over data
            for batch_idx, (inputs, labels) in enumerate(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_samples += inputs.size(0)
                
                # Print progress every 10 batches
                if (batch_idx + 1) % 10 == 0:
                    batch_acc = torch.sum(preds == labels.data).double() / inputs.size(0)
            
            # Calculate epoch metrics
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects.double() / total_samples
            
            # Store metrics in history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_accuracy'].append(epoch_acc.item())
                print(f' Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
            else:
                history['val_loss'].append(epoch_loss)
                history['val_accuracy'].append(epoch_acc.item())
                print(f' Valid Loss: {epoch_loss:.4f} | Valid Acc: {epoch_acc:.4f}')
                
                
                # Check for best model
                if epoch_acc > best_val_acc:
                    best_val_acc = epoch_acc
                    best_val_loss = epoch_loss
                    best_model_wts = copy.deepcopy(model.state_dict())
                    epochs_no_improve = 0
                    print(f'NEW BEST! Val Acc: {epoch_acc:.6f}')
                else:
                    epochs_no_improve += 1
                    print(f' No improvement for {epochs_no_improve} epochs')
        
        epoch_time = time.time() - epoch_start
        print(f'⏱️  Epoch completed in {epoch_time:.1f}s')
        
        # Early stopping check
        if epochs_no_improve >= patience:
            print(f'\n⏰ EARLY STOPPING: No improvement for {patience} epochs')
            break
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Training summary
    total_time = time.time() - start_time
    print(f'⏱️  Total time: {total_time:.1f}s ({total_time/60:.1f}m)')
    avg_epoch_time = round(total_time / len(history["train_loss"]))
    print(f'⏱️  Average time per epoch: {avg_epoch_time}s')
    Actual_epochs = len(history["train_loss"])
    print(f'Actual epochs: {Actual_epochs}')
    print(f' Best Val Accuracy: {best_val_acc:.6f}')
    print(f' Best Val Loss: {best_val_loss:.6f}')
    print(f' Epochs trained: {len(history["train_loss"])}')
    
    return model, history, Actual_epochs , avg_epoch_time

def build_custom_cnn(input_shape, num_classes):
    """
    Build CNN model
    """
    model = Sequential(name='CNN')
    # Chuẩn hóa đầu vào
    model.add(Conv2D(64, (3,3), activation='relu', input_shape=input_shape)),
    model.add(BatchNormalization()),
    model.add(MaxPooling2D(2, 2)),
    model.add(Conv2D(128, (3,3), activation='relu')),
    model.add(BatchNormalization()),
    model.add(MaxPooling2D(2,2)),
    model.add(Conv2D(256, (3,3), activation='relu')),
    model.add(BatchNormalization()),
    model.add(MaxPooling2D(2,2)),
    model.add(Conv2D(512, (3,3), activation='relu')),
    model.add(BatchNormalization()),
    model.add(MaxPooling2D(2,2)),
    model.add(Conv2D(1024, (3,3), activation='relu')),
    model.add(BatchNormalization()),
    model.add(MaxPooling2D(2,2)),
    model.add(Flatten()),
    model.add(Dense(1024, activation='relu')),
    model.add(BatchNormalization()),
    model.add(Dense(num_classes, activation='softmax'))
    return model


def list_available_resources():
    """
    return list models and dataset available
    """
    return {
        "datasets": [
            "Bag_Classes", "Bag4Classes", "Data_real", "dataset_capstone_9",
            "Drinking_waste_classification", "Garbage_classification_2",
            "garbage_classification_3", "Garbage_classification_dataset",
            "garbage_classification_enhanced", "garbage_dataset",
            "garbage1_dataset", "trash_dataset", "trashify-image-dataset",
            "TrashType_Image_Dataset", "waste_dataset"
        ],
        "models": [
            "EfficientNetB0", "Xception",
            "ResNet50", "ResNet101",
            "ResNet152", "DenseNet121",
            "VGG16", "ConvNeXtBase",
            "CNN", "ViT_B16" ,
            "EfficientNetV2L" , "CoAtNet"
        ]
    }

def download_and_extract_zips(selected_datasets, download_dir='datasets', extract_dir='data'):
    os.makedirs(download_dir, exist_ok=True)
    extracted_paths = []
    base_url = "http://clouds.iec-uit.com/smartbin.dataloader/"
    for dataset in selected_datasets:
        fname = f"{dataset}.zip"
        archive = tf.keras.utils.get_file(
            fname=fname,
            cache_dir=download_dir,
            origin=base_url + fname,
            extract=False
        )
        target_dir = os.path.join(extract_dir, dataset)
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(archive, 'r') as zip_ref:
            zip_ref.extractall(target_dir)
        # flatten nested folder
        inner = os.path.join(target_dir, dataset)
        if os.path.isdir(inner):
            for item in os.listdir(inner):
                os.rename(os.path.join(inner, item), os.path.join(target_dir, item))
            os.rmdir(inner)
        extracted_paths.append(Path(target_dir))
    return extracted_paths


def load_dataset(
    dataset_path,
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    validation_split=0.1,
    test_split=0.2,
    num_workers=2
):
    """
    Load images from folder, split into train/val/test and return generators.

    Returns: train_gen, val_gen, test_gen, train_df, val_df, test_df
    """
    data_dir = Path(dataset_path)
    files = []
    for ext in ['jpg', 'jpeg', 'png', 'JPG', 'PNG']:
        files += list(data_dir.rglob(f'*.{ext}'))
    df = pd.DataFrame({
        'filepath': [str(f) for f in files],
        'label': [f.parent.name for f in files]
    })
    # split train/test
    train_val_df, test_df = train_test_split(
        df, test_size=test_split, shuffle=shuffle, random_state=seed
    )
    relative_val = validation_split / (1.0 - test_split)
    train_df, val_df = train_test_split(
        train_val_df, test_size=relative_val, shuffle=shuffle, random_state=seed
    )
    # generators
    datagen = ImageDataGenerator(validation_split=validation_split)
    train_gen = datagen.flow_from_dataframe(
        train_df, x_col='filepath', y_col='label',
        target_size=target_size, color_mode=color_mode,
        class_mode=class_mode, batch_size=batch_size,
        shuffle=shuffle, subset='training', seed=seed
    )
    val_gen = datagen.flow_from_dataframe(
        train_df, x_col='filepath', y_col='label',
        target_size=target_size, color_mode=color_mode,
        class_mode=class_mode, batch_size=batch_size,
        shuffle=shuffle, subset='validation', seed=seed
    )
    test_gen = ImageDataGenerator().flow_from_dataframe(
        test_df, x_col='filepath', y_col='label',
        target_size=target_size, color_mode=color_mode,
        class_mode=class_mode, batch_size=batch_size,
        shuffle=False
    )
        #dataloaders
    
    all_labels = pd.concat([train_df['label'], val_df['label'], test_df['label']]).unique()
    label_encoder = LabelEncoder()
    label_encoder.fit(all_labels)
    
    # Define transforms
    train_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize(target_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    datasets_dict = {}
    datasets_dict['train'] = CustomImageDataset(
        train_df, transform=train_transform, label_encoder=label_encoder
    )
    datasets_dict['valid'] = CustomImageDataset(
        val_df, transform=val_test_transform, label_encoder=label_encoder
    )
    datasets_dict['test'] = CustomImageDataset(
        test_df, transform=val_test_transform, label_encoder=label_encoder
    )
    classes = label_encoder.classes_.tolist()
    
    # create dataloaders
    dataloaders = {}
    dataloaders['train'] = DataLoader(
        datasets_dict['train'],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    dataloaders['valid'] = DataLoader(
        datasets_dict['valid'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    dataloaders['test'] = DataLoader(
        datasets_dict['test'],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    return classes, dataloaders, train_gen, val_gen, test_gen, train_df, val_df, test_df


def train_image_classifier(
    train_dataset,
    val_dataset,
    dataloaders,
    num_classes: int,
    model_name: str = "EfficientNetB0",
    input_shape: tuple = (224, 224, 3),
    pooling: str = "avg",
    weights: str = "imagenet",
    fine_tune: bool = False,
    learning_rate: float = 1e-3,
    epochs: int = 200,
    patience: int = 50,
    checkpoint_path: str = "model_checkpoint",
    log_root: str = "training_logs",
    experiment_name: str = "image_classification"
):
    # ensure checkpoint format
    if not checkpoint_path.endswith('.weights.h5'):
        checkpoint_path += '.weights.h5'

    if model_name == 'ViT_B16' or model_name == 'CoAtNet' :
       model_name1 = ''
       if model_name == 'ViT_B16':
           model_name1 = 'vit_base_patch16_224'
       else:
           model_name1 = 'coatnet_2_rw_224.sw_in12k'
       device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
       freeze_backbone = not fine_tune
       model = setup_timm_model(model_name=model_name1, num_classes=num_classes, device=device, freeze_backbone=freeze_backbone)

        # Loss function và optimizer
       criterion = nn.CrossEntropyLoss()
       optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        # Training info
       print("STARTING TRAINING")
       print("=" * 60)
       print(f" Model: {model_name}")
       print(f" Device: {device}")
       print(f" Epochs: {epochs}")
       print(f" Patience: {patience}")
       print("=" * 60)
       trained_model, history , Actual_epochs , avg_epoch_time = train_model_optimized(model, criterion, optimizer, dataloaders, device, num_epochs=epochs, patience=patience)

       return trained_model, history , Actual_epochs , avg_epoch_time 
          
    if model_name == 'CNN':
        model = build_custom_cnn(input_shape, num_classes)
        model.compile(
            optimizer=Adam(learning_rate),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        # callbacks
        log_dir = os.path.join(log_root, experiment_name, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
        
        callbacks = [
            TensorBoard(log_dir=log_dir),
            EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', save_best_only=True)
        ]
        # Training info
        print("STARTING TRAINING")
        print("=" * 60)
        print(f" Model: {model_name}")
        print(f" Epochs: {epochs}")
        print(f" Patience: {patience}")
        print("=" * 60)
        start_time = time.time()
        history = model.fit(
            train_dataset, validation_data=val_dataset,
            epochs=epochs, callbacks=callbacks, verbose=1
        )
        total_time = time.time() - start_time
        avg_epoch_time = round(total_time / len(history.history["val_loss"]))
        print(f'⏱️  Average time per epoch: {avg_epoch_time}s')
        Actual_epochs = len(history.history["val_loss"])
        print(f'Actual epochs: {Actual_epochs}')
        
        return model, history, Actual_epochs , avg_epoch_time

    # map models
    backbones = {
        "EfficientNetB0": (EfficientNetB0, efficientnet_preprocess),
        "Xception": (Xception, xception_preprocess),
        "ResNet50": (ResNet50, resnet_preprocess),
        "ResNet101": (ResNet101, resnet_preprocess),
        "ResNet152": (ResNet152, resnet_preprocess),
        "DenseNet121": (DenseNet121, densenet_preprocess),
        "VGG16": (VGG16, vgg16_preprocess),
        "ConvNeXtBase": (ConvNeXtBase, convnext_preprocess),
        "EfficientNetV2L": (EfficientNetV2L, efficientnet_v2_preprocess)
    }
    backbone_cls, preprocess_fn = backbones.get(model_name)
    # build model
    preprocess_layer = Sequential([
        layers.Lambda(preprocess_fn, name='preprocess_input'),
        RandomFlip('horizontal'), RandomRotation(0.1), RandomZoom(0.1)
    ], name='augment')
    backbone = backbone_cls(
        input_shape=input_shape, include_top=False, weights=weights, pooling=pooling
    )
    backbone.trainable = fine_tune
    inputs = Input(shape=input_shape)
    x = preprocess_layer(inputs)
    x = backbone(x, training=fine_tune)
    # Fine-tuned Head
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    outputs = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    model.compile(
        optimizer=Adam(learning_rate),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    # callbacks
    log_dir = os.path.join(log_root, experiment_name, datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    callbacks = [
        TensorBoard(log_dir=log_dir),
        EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, monitor='val_accuracy', save_best_only=True)
    ]
     # Training info
    print("STARTING TRAINING")
    print("=" * 60)
    print(f" Model: {model_name}")
    print(f" Epochs: {epochs}")
    print(f" Patience: {patience}")
    print("=" * 60)
    start_time = time.time()
    history = model.fit(
        train_dataset, validation_data=val_dataset,
        epochs=epochs, callbacks=callbacks, verbose=1
    )
    total_time = time.time() - start_time
    avg_epoch_time = round(total_time / len(history.history["val_loss"]))
    print(f'⏱️  Average time per epoch: {avg_epoch_time}s')
    Actual_epochs = len(history.history["val_loss"])
    print(f'Actual epochs: {Actual_epochs}')
    
    return model, history, Actual_epochs , avg_epoch_time


def evaluate_tensorflow(model, history, test_dataset, test_df, train_generator, classes, model_name, actual_epochs, patience, avg_epoch_time):
    # evaluate
    loss, acc = model.evaluate(test_dataset, verbose=0)
    print(f"Test Loss: {loss:.5f}")
    print(f"Test Accuracy: {acc*100:.2f}%")

    # predict
    preds = model.predict(test_dataset)
    pred_labels = np.argmax(preds, axis=1)
    mapping = {v:k for k,v in train_generator.class_indices.items()}
    pred_names = [mapping[i] for i in pred_labels]
    y_true = list(test_df.label)

    # Create classification report
    report = classification_report(y_true, pred_names, target_names=classes, digits=4)
    print("\n📊 Classification Report on Test Set:")
    print(report)

        # Calculate metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, pred_names, average='weighted')
    accuracy = accuracy_score(y_true, pred_names)
    
    print(f"📊 Overall Test Accuracy: {accuracy:.6f}")
    
    metrics = [model_name,actual_epochs,patience,round(precision * 100, 2), round(recall * 100, 2), round(f1 * 100, 2) , round(accuracy * 100, 2), round(avg_epoch_time, 0)]


    return metrics

def evaluate_model_pytorch(model, test_loader, classes, model_name, actual_epochs, patience, avg_epoch_time):
    """
    Evaluate model trên test set và trả về các metrics dạng dictionary
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    running_corrects = 0
    total_samples = 0
    class_correct = list(0. for i in range(len(classes)))
    class_total = list(0. for i in range(len(classes)))
    
    print("\n🧪 EVALUATING ON TEST SET")
    print("-" * 40)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            # Collect all predictions and labels for detailed metrics
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Per-class accuracy
            c = (preds == labels).squeeze()
            for i in range(labels.size(0)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    # Calculate metrics
    test_acc = running_corrects.double() / total_samples
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    accuracy = accuracy_score(all_labels, all_preds)
    
    print(f"📊 Overall Test Accuracy: {test_acc:.6f}")
    
    # Per-class accuracy
    print("\n📊 Per-class Accuracy:")
    for i in range(len(classes)):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
            print(f"  {classes[i]}: {acc:.2f}% ({int(class_correct[i])}/{int(class_total[i])})")
    
    # Tạo classification report
    report = classification_report(all_labels, all_preds, target_names=classes, digits=4)
    print("\n📊 Classification Report on Test Set:")
    print(report)

    metrics = [model_name,actual_epochs,patience,round(precision * 100, 2), round(recall * 100, 2), round(f1 * 100, 2) , round(accuracy * 100, 2), round(avg_epoch_time, 0)]


    return metrics

def plot_training_history(history, model_name=None):
    """
    Plot training history for a single model
    """
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Plot Loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Validation Loss')
    
    title = 'Loss over Epochs'
    if model_name:
        title = f'{model_name} - Loss over Epochs'
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['train_accuracy'], label='Train Accuracy')
    plt.plot(epochs, history['val_accuracy'], label='Validation Accuracy')
    
    title = 'Accuracy over Epochs'
    if model_name:
        title = f'{model_name} - Accuracy over Epochs'
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_multiple_training_histories(histories_dict, figsize=(15, 10)):
    """
    Plot training histories for multiple models on the same chart
    
    Args:
        histories_dict: Dictionary with model names as keys and histories as values
        Example: {
            'EfficientNetB0': history1,
            'CNN': history2, 
            'Xception': history3,
            ...
        }
        figsize: Figure size tuple
    """
    
    # Define colors for different models
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))
    
    plt.figure(figsize=figsize)
    
    # Plot Training and Validation Loss
    plt.subplot(2, 2, 1)
    for i, (model_name, history) in enumerate(histories_dict.items()):
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], 
                color=colors[i], linestyle='-', alpha=0.8,
                label=f'{model_name}')
    
    plt.title('Training Loss Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot Validation Loss
    plt.subplot(2, 2, 2)
    for i, (model_name, history) in enumerate(histories_dict.items()):
        epochs = range(1, len(history['val_loss']) + 1)
        plt.plot(epochs, history['val_loss'], 
                color=colors[i], linestyle='--', alpha=0.8,
                label=f'{model_name}')
    
    plt.title('Validation Loss Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot Training Accuracy
    plt.subplot(2, 2, 3)
    for i, (model_name, history) in enumerate(histories_dict.items()):
        epochs = range(1, len(history['train_accuracy']) + 1)
        plt.plot(epochs, history['train_accuracy'], 
                color=colors[i], linestyle='-', alpha=0.8,
                label=f'{model_name}')
    
    plt.title('Training Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot Validation Accuracy
    plt.subplot(2, 2, 4)
    for i, (model_name, history) in enumerate(histories_dict.items()):
        epochs = range(1, len(history['val_accuracy']) + 1)
        plt.plot(epochs, history['val_accuracy'], 
                color=colors[i], linestyle='--', alpha=0.8,
                label=f'{model_name}')
    
    plt.title('Validation Accuracy Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_combined_training_histories(histories_dict, figsize=(14, 6)):
    """
    Plot training histories for multiple models in 2 subplots (Loss and Accuracy)
    Each subplot shows both train and validation curves for all models
    
    Args:
        histories_dict: Dictionary with model names as keys and histories as values
        figsize: Figure size tuple
    """
    
    # Define colors and line styles
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))
    
    plt.figure(figsize=figsize)
    
    # Plot Loss (Train and Validation together)
    plt.subplot(1, 2, 1)
    for i, (model_name, history) in enumerate(histories_dict.items()):
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], 
                color=colors[i], linestyle='-', alpha=0.8,
                label=f'{model_name} (Train)')
        plt.plot(epochs, history['val_loss'], 
                color=colors[i], linestyle='--', alpha=0.8,
                label=f'{model_name} (Val)')
    
    plt.title('Loss Comparison - All Models', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot Accuracy (Train and Validation together)
    plt.subplot(1, 2, 2)
    for i, (model_name, history) in enumerate(histories_dict.items()):
        epochs = range(1, len(history['train_accuracy']) + 1)
        plt.plot(epochs, history['train_accuracy'], 
                color=colors[i], linestyle='-', alpha=0.8,
                label=f'{model_name} (Train)')
        plt.plot(epochs, history['val_accuracy'], 
                color=colors[i], linestyle='--', alpha=0.8,
                label=f'{model_name} (Val)')
    
    plt.title('Accuracy Comparison - All Models', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_training_plots(histories_dict, save_path='training_comparison.png', figsize=(14, 6)):
    """
    Save the training comparison plots to file
    """
    colors = plt.cm.tab10(np.linspace(0, 1, len(histories_dict)))
    plt.figure(figsize=figsize)
    
    # Plot Loss
    plt.subplot(1, 2, 1)
    for i, (model_name, history) in enumerate(histories_dict.items()):
        epochs = range(1, len(history['train_loss']) + 1)
        plt.plot(epochs, history['train_loss'], 
                color=colors[i], linestyle='-', alpha=0.8,
                label=f'{model_name} (Train)')
        plt.plot(epochs, history['val_loss'], 
                color=colors[i], linestyle='--', alpha=0.8,
                label=f'{model_name} (Val)')
    
    plt.title('Loss Comparison - All Models', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot Accuracy
    plt.subplot(1, 2, 2)
    for i, (model_name, history) in enumerate(histories_dict.items()):
        epochs = range(1, len(history['train_accuracy']) + 1)
        plt.plot(epochs, history['train_accuracy'], 
                color=colors[i], linestyle='-', alpha=0.8,
                label=f'{model_name} (Train)')
        plt.plot(epochs, history['val_accuracy'], 
                color=colors[i], linestyle='--', alpha=0.8,
                label=f'{model_name} (Val)')
    
    plt.title('Accuracy Comparison - All Models', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
# change name history
def standardize_keys(metrics_dict):
    new_dict = {}
    for key, value in metrics_dict.items():
        if key == 'loss':
            new_dict['train_loss'] = value
        elif key == 'accuracy':
            new_dict['train_accuracy'] = value
        else:
            new_dict[key] = value
    return new_dict
    
def benchmark_model(
    models_to_train,
    dataset_path,
    input_shape=(224, 224, 3),
    target_size=(224, 224),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=32,
    shuffle=True,
    seed=42,
    validation_split=0.1,
    test_split=0.2,
    num_workers=2,
    epochs=5,
    patience=50,
    learning_rate=1e-3,
    fine_tune=False,
    checkpoint_path='model_checkpoint'
):
    """
    Train and evaluate multiple image classification models.

    Parameters:
        models_to_train (list): List of model names to train.
        dataset_path (str): Path to dataset directory.
        input_shape (tuple): Input shape for the models.
        target_size (tuple): Image resize dimensions.
        color_mode (str): 'rgb' or 'grayscale'.
        class_mode (str): 'categorical', 'binary', etc.
        batch_size (int): Batch size.
        shuffle (bool): Shuffle training data.
        seed (int): Random seed.
        validation_split (float): Ratio for validation.
        test_split (float): Ratio for test set.
        num_workers (int): Number of data loading workers.
        epochs (int): Max training epochs.
        patience (int): Early stopping patience.
        learning_rate (float): Learning rate.
        fine_tune (bool): Whether to fine-tune pretrained layers.
        checkpoint_path (str): Path to save model checkpoints.

    Returns:
        pd.DataFrame: DataFrame containing model performance metrics.
    """

    pytorch_timm_model = ['CoAtNet', 'ViT_B16']
    tensorflow_model = ['CNN', 'VGG16', 'DenseNet121', 'ResNet152',
                        'ResNet101', 'ResNet50', 'Xception', 'EfficientNetV2L']
    column_order = ['Mô hình', 'Số vòng lặp thực tế', 'Patience', 'Precision (%)', 
                    'Recall (%)', 'F1-score (%)', 'Accuracy (%)', 'Thời gian/epoch (s)']

    classes, dataloaders, train_gen, val_gen, test_gen, train_df, val_df, test_df = load_dataset(
        dataset_path,
        target_size,
        color_mode,
        class_mode,
        batch_size,
        shuffle,
        seed,
        validation_split,
        test_split,
        num_workers
    )

    all_histories = {}
    all_metrics = pd.DataFrame(columns=column_order)

    for model_name in models_to_train:
        model, history, actual_epochs, avg_epoch_time = train_image_classifier(
            train_gen,
            val_gen,
            dataloaders=dataloaders,
            num_classes=len(classes),
            model_name=model_name,
            input_shape=input_shape,
            pooling='avg',
            weights='imagenet',
            fine_tune=fine_tune,
            learning_rate=learning_rate,
            epochs=epochs,
            patience=patience,
            checkpoint_path=checkpoint_path,
            log_root='training_logs',
            experiment_name='image_classification'
        )

        if model_name in pytorch_timm_model:
            all_histories[model_name] = history
            metric = evaluate_model_pytorch(
                model, dataloaders['test'], classes,
                model_name, actual_epochs, patience, avg_epoch_time
            )
        else:
            all_histories[model_name] = history.history
            metric = evaluate_tensorflow(
                model, history, test_gen, test_df, train_gen, classes,
                model_name, actual_epochs, patience, avg_epoch_time
            )

        all_metrics.loc[len(all_metrics)] = metric

        # cleanup
        del model, history, actual_epochs, avg_epoch_time, metric
        
    # Chuẩn hóa khóa để trực quan hóa
    for model_name in all_histories:
        all_histories[model_name] = standardize_keys(all_histories[model_name])

    print("🎨 Plotting training comparisons...")
    plot_multiple_training_histories(all_histories)
    plot_combined_training_histories(all_histories)

    return all_metrics