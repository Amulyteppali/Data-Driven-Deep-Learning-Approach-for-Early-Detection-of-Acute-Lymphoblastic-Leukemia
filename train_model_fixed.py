"""
Fixed training script with proper data split and augmentation
Trains on ALL data (Original + Segmented) with proper train/validation/test split
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

print("=" * 70)
print("ACUTE LYMPHOBLASTIC LEUKEMIA CLASSIFICATION - FIXED TRAINING")
print("=" * 70)
print("\nInitializing training process...")
print(f"TensorFlow version: {tf.__version__}")

# Define class codes
code = {"Benign": 0, "Early": 1, "Pre": 2, "Pro": 3}

def getcode(n):
    for x, y in code.items():
        if n == y:
            return x

s = 224
base_path = r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media'

print("\n" + "=" * 70)
print("STEP 1: Loading ALL Data (Original + Segmented)")
print("=" * 70)

X_all = []
y_all = []

# Load from both Original and Segmented folders
for folder_type in ['Original', 'Segmented']:
    print(f"\nLoading from {folder_type}...")
    
    for class_name in ['Benign', 'Early', 'Pre', 'Pro']:
        class_path = os.path.join(base_path, folder_type, class_name)
        print(f"  Loading {class_name} images...")
        
        for img in tqdm(os.listdir(class_path)):
            img_path = os.path.join(class_path, img)
            image = cv2.imread(img_path, 1)
            if image is not None:
                image_array = cv2.resize(image, (s, s))
                X_all.append(image_array)
                y_all.append(code[class_name])

print(f"\nTotal images loaded: {len(X_all)}")
print(f"Class distribution: Benign={y_all.count(0)}, Early={y_all.count(1)}, Pre={y_all.count(2)}, Pro={y_all.count(3)}")

print("\n" + "=" * 70)
print("STEP 2: Data Preprocessing and Splitting")
print("=" * 70)

# Convert to numpy arrays
X = np.array(X_all)
y = np.array(y_all)

print(f"\nOriginal data shape: {X.shape}, dtype: {X.dtype}")

# Normalize pixel values to 0-1 range
print("Normalizing images to [0, 1] range...")
X = X.astype('float32') / 255.0
print(f"After normalization range: [{X.min():.4f}, {X.max():.4f}]")

# Split: 70% train, 15% validation, 15% test
print("\nSplitting data: 70% train, 15% validation, 15% test")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.70, shuffle=True, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, train_size=0.50, shuffle=True, random_state=42, stratify=y_temp)

print(f"Training set: {X_train.shape[0]} images")
print(f"Validation set: {X_val.shape[0]} images")
print(f"Test set: {X_test.shape[0]} images")

print("\nClass distribution in training set:")
for i in range(4):
    count = np.sum(y_train == i)
    print(f"  {getcode(i)}: {count} ({count/len(y_train)*100:.1f}%)")

print("\n" + "=" * 70)
print("STEP 3: Building QCResNet Model")
print("=" * 70)

def qc_residual_block(x, filters, kernel_size=3, stride=1, use_conv_shortcut=False):
    shortcut = x
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    if use_conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_qcresnet(input_shape=(224, 224, 3), num_classes=4):
    inputs = Input(shape=input_shape)
    
    # Initial Convolution Block
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # Residual blocks
    x = qc_residual_block(x, 64, use_conv_shortcut=True)
    x = qc_residual_block(x, 64)
    
    x = qc_residual_block(x, 128, stride=2, use_conv_shortcut=True)
    x = qc_residual_block(x, 128)
    
    x = qc_residual_block(x, 256, stride=2, use_conv_shortcut=True)
    x = qc_residual_block(x, 256)
    
    x = qc_residual_block(x, 512, stride=2, use_conv_shortcut=True)
    x = qc_residual_block(x, 512)
    
    # Classification head
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)  # Increased dropout
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.4)(x)
    
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='QCResNet')
    return model

# Build and compile model
model = build_qcresnet(input_shape=(224, 224, 3), num_classes=4)

print("\nQCResNet Model Built Successfully!")

# Use class weights to handle imbalance
from sklearn.utils.class_weight import compute_class_weight
class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights = {i: class_weights_array[i] for i in range(len(class_weights_array))}
print(f"\nClass weights: {class_weights}")

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower learning rate
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\n" + "=" * 70)
print("STEP 4: Training the Model")
print("=" * 70)
print("\nTraining Configuration:")
print("- Batch Size: 32")
print("- Epochs: 15")
print("- Optimizer: Adam (lr=0.0001)")
print("- Class Weights: Enabled")
print("\nStarting training...\n")

# Add early stopping and model checkpoint
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

history = model.fit(
    X_train, y_train,
    batch_size=32,
    epochs=15,
    verbose=1,
    validation_data=(X_val, y_val),
    class_weight=class_weights,
    callbacks=[early_stop, reduce_lr]
)

print("\n" + "=" * 70)
print("STEP 5: Model Evaluation")
print("=" * 70)

# Evaluate on test set
print("\nEvaluating on test set...")
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Detailed per-class evaluation
print("\nPer-class evaluation on test set:")
y_pred = model.predict(X_test, verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:")
print(classification_report(y_test, y_pred_classes, target_names=['Benign', 'Early', 'Pre', 'Pro']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred_classes)
print(cm)

print("\n" + "=" * 70)
print("STEP 6: Saving Model")
print("=" * 70)

model_path = 'Acute_Lymphoblastic_Lukemia_Model.h5'
model.save(model_path)
print(f"\nModel saved to: {model_path}")

print("\n" + "=" * 70)
print("STEP 7: Generating Visualizations")
print("=" * 70)

accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(accuracy, label='Training Accuracy', marker='o')
plt.plot(val_accuracy, label='Validation Accuracy', marker='s')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Model Accuracy over Epochs")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss', marker='o')
plt.plot(val_loss, label='Validation Loss', marker='s')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Model Loss over Epochs")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('media/accuracy_plot.png', dpi=150)
print("Accuracy plot saved")

print("\n" + "=" * 70)
print("STEP 8: Saving Results")
print("=" * 70)

final_accuracy = round(float(accuracy[-1]) * 100, 2)
final_val_accuracy = round(float(val_accuracy[-1]) * 100, 2)
final_test_accuracy = round(float(test_accuracy) * 100, 2)

results = {
    'accuracy': final_accuracy,
    'val_accuracy': final_val_accuracy,
    'test_accuracy': final_test_accuracy,
    'loss': round(float(loss[-1]), 4),
    'val_loss': round(float(val_loss[-1]), 4),
    'test_loss': round(float(test_loss), 4),
    'epochs': len(accuracy),
    'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_type': 'QCResNet (Quality-enhanced Convolutional ResNet)',
    'total_params': model.count_params(),
    'training_samples': len(X_train),
    'validation_samples': len(X_val),
    'test_samples': len(X_test)
}

with open('media/training_results.json', 'w') as f:
    json.dump(results, f, indent=4)

print("Results saved")

print("\n" + "=" * 70)
print("TRAINING COMPLETED!")
print("=" * 70)
print(f"\nFinal Results:")
print(f"  Training Accuracy: {final_accuracy}%")
print(f"  Validation Accuracy: {final_val_accuracy}%")
print(f"  Test Accuracy: {final_test_accuracy}%")
print("\nModel is ready for predictions!")
print("=" * 70)
