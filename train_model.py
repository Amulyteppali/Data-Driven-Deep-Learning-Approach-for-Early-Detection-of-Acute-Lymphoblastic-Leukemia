"""
Standalone script to train the Acute Lymphoblastic Leukemia Classification Model
This script trains the QCResNet model with proper preprocessing
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import tensorflow as tf
from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D
from keras.layers import GlobalAveragePooling2D, Dense, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

print("=" * 70)
print("ACUTE LYMPHOBLASTIC LEUKEMIA CLASSIFICATION - MODEL TRAINING")
print("=" * 70)
print("\nInitializing training process...")
print(f"TensorFlow version: {tf.__version__}")
print(f"Keras backend: {tf.keras.backend.backend()}")

# Define class codes
code = {"Benign": 0, "Early": 1, "Pre": 2, "Pro": 3}

def getcode(n):
    for x, y in code.items():
        if n == y:
            return x

s = 224
base_path = r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media'

print("\n" + "=" * 70)
print("STEP 1: Loading Training Data (Original Images)")
print("=" * 70)

X_train = []
y_train = []

# Load Benign images
print("\nLoading Benign images...")
benign_path = os.path.join(base_path, 'Original', 'Benign')
for img in tqdm(os.listdir(benign_path)):
    img_path = os.path.join(benign_path, img)
    image = cv2.imread(img_path, 1)
    if image is not None:
        image_array = cv2.resize(image, (s, s))
        X_train.append(image_array)
        y_train.append(code['Benign'])

# Load Early images
print("Loading Early images...")
early_path = os.path.join(base_path, 'Original', 'Early')
for img in tqdm(os.listdir(early_path)):
    img_path = os.path.join(early_path, img)
    image = cv2.imread(img_path, 1)
    if image is not None:
        image_array = cv2.resize(image, (s, s))
        X_train.append(image_array)
        y_train.append(code['Early'])

# Load Pre images
print("Loading Pre images...")
pre_path = os.path.join(base_path, 'Original', 'Pre')
for img in tqdm(os.listdir(pre_path)):
    img_path = os.path.join(pre_path, img)
    image = cv2.imread(img_path, 1)
    if image is not None:
        image_array = cv2.resize(image, (s, s))
        X_train.append(image_array)
        y_train.append(code['Pre'])

# Load Pro images
print("Loading Pro images...")
pro_path = os.path.join(base_path, 'Original', 'Pro')
for img in tqdm(os.listdir(pro_path)):
    img_path = os.path.join(pro_path, img)
    image = cv2.imread(img_path, 1)
    if image is not None:
        image_array = cv2.resize(image, (s, s))
        X_train.append(image_array)
        y_train.append(code['Pro'])

print(f"\nTotal training images loaded: {len(X_train)}")
print(f"Class distribution: Benign={y_train.count(0)}, Early={y_train.count(1)}, Pre={y_train.count(2)}, Pro={y_train.count(3)}")

print("\n" + "=" * 70)
print("STEP 2: Loading Test Data (Segmented Images)")
print("=" * 70)

X_test = []
y_test = []

# Load Benign test images
print("\nLoading Benign test images...")
benign_test_path = os.path.join(base_path, 'Segmented', 'Benign')
for img in tqdm(os.listdir(benign_test_path)):
    img_path = os.path.join(benign_test_path, img)
    image = cv2.imread(img_path, 1)
    if image is not None:
        image_array = cv2.resize(image, (s, s))
        X_test.append(image_array)
        y_test.append(code['Benign'])

# Load Early test images
print("Loading Early test images...")
early_test_path = os.path.join(base_path, 'Segmented', 'Early')
for img in tqdm(os.listdir(early_test_path)):
    img_path = os.path.join(early_test_path, img)
    image = cv2.imread(img_path, 1)
    if image is not None:
        image_array = cv2.resize(image, (s, s))
        X_test.append(image_array)
        y_test.append(code['Early'])

# Load Pre test images
print("Loading Pre test images...")
pre_test_path = os.path.join(base_path, 'Segmented', 'Pre')
for img in tqdm(os.listdir(pre_test_path)):
    img_path = os.path.join(pre_test_path, img)
    image = cv2.imread(img_path, 1)
    if image is not None:
        image_array = cv2.resize(image, (s, s))
        X_test.append(image_array)
        y_test.append(code['Pre'])

# Load Pro test images
print("Loading Pro test images...")
pro_test_path = os.path.join(base_path, 'Segmented', 'Pro')
for img in tqdm(os.listdir(pro_test_path)):
    img_path = os.path.join(pro_test_path, img)
    image = cv2.imread(img_path, 1)
    if image is not None:
        image_array = cv2.resize(image, (s, s))
        X_test.append(image_array)
        y_test.append(code['Pro'])

print(f"\nTotal test images loaded: {len(X_test)}")
print(f"Class distribution: Benign={y_test.count(0)}, Early={y_test.count(1)}, Pre={y_test.count(2)}, Pro={y_test.count(3)}")

print("\n" + "=" * 70)
print("STEP 3: Data Preprocessing")
print("=" * 70)

# Shuffle training data
da = list(zip(X_train, y_train))
import random
random.shuffle(da)
X_train, y_train = zip(*da)

# Convert to numpy arrays
X = np.array(X_train)
y = np.array(y_train)
X_test_array = np.array(X_test)
y_test_array = np.array(y_test)

print(f"\nOriginal data shapes:")
print(f"X_train: {X.shape}, dtype: {X.dtype}")
print(f"X_test: {X_test_array.shape}, dtype: {X_test_array.dtype}")

# Normalize pixel values to 0-1 range
print("\nNormalizing images to [0, 1] range...")
X = X.astype('float32') / 255.0
X_test_array = X_test_array.astype('float32') / 255.0

print(f"After normalization:")
print(f"X_train range: [{X.min():.4f}, {X.max():.4f}]")
print(f"X_test range: [{X_test_array.min():.4f}, {X_test_array.max():.4f}]")

# Split training data
xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, shuffle=True, random_state=42)

print(f"\nData split:")
print(f"Training set: {xtrain.shape[0]} images")
print(f"Validation set: {xtest.shape[0]} images")
print(f"Test set (Segmented): {X_test_array.shape[0]} images")

print("\n" + "=" * 70)
print("STEP 4: Building QCResNet Model Architecture")
print("=" * 70)

def qc_residual_block(x, filters, kernel_size=3, stride=1, use_conv_shortcut=False):
    """
    Quality-enhanced Convolutional Residual Block
    Implements skip connections for better gradient flow
    """
    shortcut = x
    
    # First convolution layer
    x = Conv2D(filters, kernel_size, strides=stride, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second convolution layer
    x = Conv2D(filters, kernel_size, strides=1, padding='same')(x)
    x = BatchNormalization()(x)
    
    # Shortcut connection (skip connection)
    if use_conv_shortcut:
        shortcut = Conv2D(filters, 1, strides=stride, padding='same')(shortcut)
        shortcut = BatchNormalization()(shortcut)
    
    # Add shortcut to main path (residual connection)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def build_qcresnet(input_shape=(224, 224, 3), num_classes=4):
    """
    QCResNet - Quality-enhanced Convolutional ResNet
    Custom architecture for Acute Lymphoblastic Leukemia Classification
    """
    inputs = Input(shape=input_shape)
    
    # Initial Convolution Block (Quality Enhancement Stage 1)
    x = Conv2D(64, 7, strides=2, padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(3, strides=2, padding='same')(x)
    
    # QC Residual Stage 1 (64 filters)
    x = qc_residual_block(x, 64, use_conv_shortcut=True)
    x = qc_residual_block(x, 64)
    
    # QC Residual Stage 2 (128 filters)
    x = qc_residual_block(x, 128, stride=2, use_conv_shortcut=True)
    x = qc_residual_block(x, 128)
    
    # QC Residual Stage 3 (256 filters)
    x = qc_residual_block(x, 256, stride=2, use_conv_shortcut=True)
    x = qc_residual_block(x, 256)
    
    # QC Residual Stage 4 (512 filters)
    x = qc_residual_block(x, 512, stride=2, use_conv_shortcut=True)
    x = qc_residual_block(x, 512)
    
    # Quality Enhancement Classification Head
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    # Output layer for 4 classes (Benign, Early, Pre, Pro)
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs, name='QCResNet')
    return model

# Build the model
model = build_qcresnet(input_shape=(224, 224, 3), num_classes=4)

print("\nQCResNet Model Architecture Built Successfully!")
print("=" * 50)
print("QCResNet - Quality-enhanced Convolutional ResNet")
print("Custom architecture for ALL Classification")
print("=" * 50)

# Compile model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("\nModel Summary:")
model.summary()

print("\n" + "=" * 70)
print("STEP 5: Training the Model")
print("=" * 70)
print(f"\nTraining Configuration:")
print(f"- Batch Size: 128")
print(f"- Epochs: 10")
print(f"- Optimizer: Adam")
print(f"- Loss: Sparse Categorical Crossentropy")
print("\nStarting training...\n")

# Train the model
history = model.fit(
    xtrain, ytrain,
    batch_size=128,
    epochs=10,
    verbose=1,
    validation_data=(xtest, ytest)
)

print("\n" + "=" * 70)
print("STEP 6: Saving Model")
print("=" * 70)

# Save the model
model_path = 'Acute_Lymphoblastic_Lukemia_Model.h5'
model.save(model_path)
print(f"\n✓ Model saved to: {model_path}")

print("\n" + "=" * 70)
print("STEP 7: Evaluating Model Performance")
print("=" * 70)

# Evaluate on validation set
print("\nEvaluating on validation set...")
val_loss, val_accuracy = model.evaluate(xtest, ytest, verbose=0)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

# Evaluate on test set (Segmented images)
print("\nEvaluating on test set (Segmented images)...")
test_loss, test_accuracy = model.evaluate(X_test_array, y_test_array, verbose=0)
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

print("\n" + "=" * 70)
print("STEP 8: Generating Visualizations")
print("=" * 70)

# Extract training history
accuracy = history.history['accuracy']
val_accuracy = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create accuracy plot
print("\nGenerating accuracy plot...")
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
print("✓ Accuracy plot saved to: media/accuracy_plot.png")

print("\n" + "=" * 70)
print("STEP 9: Saving Training Results")
print("=" * 70)

# Prepare results
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
    'accuracy_list': [round(float(a) * 100, 2) for a in accuracy],
    'val_accuracy_list': [round(float(a) * 100, 2) for a in val_accuracy],
    'loss_list': [round(float(l), 4) for l in loss],
    'val_loss_list': [round(float(l), 4) for l in val_loss],
    'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'model_type': 'QCResNet (Quality-enhanced Convolutional ResNet)',
    'total_params': model.count_params(),
    'trainable_params': sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]),
    'training_samples': len(xtrain),
    'validation_samples': len(xtest),
    'test_samples': len(X_test_array)
}

results_path = 'media/training_results.json'
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\n✓ Training results saved to: {results_path}")

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY!")
print("=" * 70)
print("\n📊 Final Results:")
print(f"   Training Accuracy: {final_accuracy}%")
print(f"   Validation Accuracy: {final_val_accuracy}%")
print(f"   Test Accuracy (Segmented): {final_test_accuracy}%")
print(f"\n✅ Model is ready for predictions!")
print("=" * 70)
