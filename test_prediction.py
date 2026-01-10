"""
Test the trained model with a sample prediction
"""
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf
import os

print("=" * 70)
print("MODEL PREDICTION TEST")
print("=" * 70)

# Load the trained model
print("\nLoading trained model...")
model = load_model(
    'Acute_Lymphoblastic_Lukemia_Model.h5',
    custom_objects={'softmax_v2': tf.nn.softmax}
)
print("Model loaded successfully!")

# Define the label mappings
code = {"Benign": 0, "Early": 1, "Pre": 2, "Pro": 3}
reverse_code = {v: k for k, v in code.items()}

# Test with an image from each class
base_path = r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Segmented'

print("\n" + "=" * 70)
print("Testing predictions on sample images")
print("=" * 70)

for class_name in ['Benign', 'Early', 'Pre', 'Pro']:
    class_path = os.path.join(base_path, class_name)
    
    # Get the first image from this class
    images = os.listdir(class_path)
    if len(images) > 0:
        img_path = os.path.join(class_path, images[0])
        
        print(f"\n\n--- Testing {class_name} Image ---")
        print(f"Image: {images[0]}")
        
        # Read and preprocess image (exactly as in training)
        img = cv2.imread(img_path, 1)  # BGR format
        
        if img is not None:
            # Resize to 224x224
            img_resized = cv2.resize(img, (224, 224))
            
            # Normalize to 0-1 range
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Make prediction
            predictions = model.predict(img_batch, verbose=0)
            
            # Get predicted class and confidence
            predicted_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0]) * 100
            predicted_class = reverse_code[predicted_idx]
            
            print(f"\nTrue Label: {class_name}")
            print(f"Predicted: {predicted_class}")
            print(f"Confidence: {confidence:.2f}%")
            
            print(f"\nAll Class Probabilities:")
            for i, prob in enumerate(predictions[0]):
                print(f"  {reverse_code[i]}: {prob * 100:.2f}%")
            
            # Check if prediction is correct
            if predicted_class == class_name:
                print("\nResult: CORRECT PREDICTION!")
            else:
                print(f"\nResult: INCORRECT - Expected {class_name}, got {predicted_class}")

print("\n" + "=" * 70)
print("PREDICTION TEST COMPLETED")
print("=" * 70)
