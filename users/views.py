import re
from django.shortcuts import render
from .forms import UserRegistrationForm
from django.contrib import messages
from .models import UserRegistrationModel
import pandas as pd
import csv


# Create your views here.
def UserRegisterActions(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            print('Data is Valid')
            form.save()
            messages.success(request, 'You have been successfully registered')
            form = UserRegistrationForm()
            return render(request, 'UserRegistrations.html', {'form': form})
        else:
            messages.success(request, 'Email or Mobile Already Existed')
            print("Invalid form")
    else:
        form = UserRegistrationForm()
    return render(request, 'UserRegistrations.html', {'form': form})


def UserLoginCheck(request):
    if request.method == "POST":
        loginid = request.POST.get('loginid')
        pswd = request.POST.get('pswd')
        print("Login ID = ", loginid, 'Password = ', pswd)
        try:
            check = UserRegistrationModel.objects.get(loginid=loginid, password=pswd)
            status = check.status
            print('Status is = ', status)
            if status == "activated":
                request.session['id'] = check.id
                request.session['loggeduser'] = check.name
                request.session['loginid'] = loginid
                request.session['email'] = check.email
                print("User id At", check.id, status)
                return render(request, 'users/UserHome.html', {})
            else:
                messages.success(request, 'Your Account Not at activated')
                return render(request, 'UserLogin.html')
        except Exception as e:
            print('Exception is ', str(e))
            pass
        messages.success(request, 'Invalid Login id and password')
    return render(request, 'UserLogin.html', {})


def UserHome(request):
    return render(request, 'users/UserHome.html', {})

def training(request):
    import os
    import json
    
    # Check if this is just a GET request without training
    if request.method == 'GET' and 'start' not in request.GET:
        # Check if we have saved accuracy results
        accuracy_file = 'media/training_results.json'
        model_file = 'Acute_Lymphoblastic_Lukemia_Model.h5'
        
        if os.path.exists(accuracy_file):
            with open(accuracy_file, 'r') as f:
                results = json.load(f)
            return render(request, 'users/training.html', results)
        elif os.path.exists(model_file):
            # Model exists but no saved results - evaluate the model to get accuracy
            import numpy as np
            import cv2
            from keras.models import load_model
            import tensorflow as tf
            
            try:
                # Load the trained model
                model = load_model(model_file, custom_objects={'softmax_v2': tf.nn.softmax})
                
                code = {"Benign": 0, "Early": 1, "Pre": 2, "Pro": 3}
                s = 224
                
                X_test = []
                y_test = []
                
                # Load test images from Segmented folder
                base_path = r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Segmented'
                
                for class_name in ['Benign', 'Early', 'Pre', 'Pro']:
                    class_path = os.path.join(base_path, class_name)
                    if os.path.exists(class_path):
                        for img_name in os.listdir(class_path):
                            img_path = os.path.join(class_path, img_name)
                            image = cv2.imread(img_path, 1)
                            if image is not None:
                                image_array = cv2.resize(image, (s, s))
                                X_test.append(image_array)
                                y_test.append(code[class_name])
                
                if len(X_test) > 0:
                    X_test = np.array(X_test)
                    y_test = np.array(y_test)
                    
                    # Normalize pixel values to 0-1 range (same as training)
                    X_test = X_test.astype('float32') / 255.0
                    
                    # Evaluate the model
                    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
                    accuracy_percent = round(accuracy * 100, 2)
                    
                    # Save results for future use
                    results = {
                        'accuracy': accuracy_percent,
                        'val_accuracy': accuracy_percent,
                        'model_exists': True
                    }
                    with open(accuracy_file, 'w') as f:
                        json.dump(results, f)
                    
                    return render(request, 'users/training.html', results)
                else:
                    return render(request, 'users/training.html', {
                        'message': 'Model exists but no test images found to evaluate.',
                        'model_exists': True,
                        'show_retrain_button': True
                    })
            except Exception as e:
                return render(request, 'users/training.html', {
                    'message': f'Model exists but could not be evaluated: {str(e)}',
                    'model_exists': True,
                    'show_retrain_button': True
                })
        else:
            return render(request, 'users/training.html', {
                'message': 'Click the button below to start training',
                'show_start_button': True
            })
    
    # Import all required libraries for training
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    sns.set(style="whitegrid")
    import glob as gb
    import tensorflow as tf
    import keras
    import cv2
    from keras.src.legacy.preprocessing.image import ImageDataGenerator
    from keras.utils import load_img, img_to_array
    from keras import utils as image
    
    print("Starting training process...")
    from keras.utils import to_categorical
    
    code = {"Benign":0 ,"Early":1,"Pre":2,"Pro":3}

    def getcode(n) :
        for x , y in code.items() :
            if n == y :
                return x
    
    s=224
    import cv2
    from tqdm import tqdm
    import os
    
    X_train = []
    y_train = []
    for img in tqdm(os.listdir(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Original\Benign')):
        image = cv2.imread(os.path.join(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Original\Benign',img),1)
        image_array = cv2.resize(image , (s,s))
        X_train.append(list(image_array))
        y_train.append(code['Benign'])
        
    for img in tqdm(os.listdir(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Original\Early')):
        image = cv2.imread(os.path.join(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Original\Early',img),1)
        image_array = cv2.resize(image , (s,s))
        X_train.append(list(image_array))
        y_train.append(code['Early'])
        
    for img in tqdm(os.listdir(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Original\Pre')):
        image = cv2.imread(os.path.join(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Original\Pre',img),1)
        image_array = cv2.resize(image , (s,s))
        X_train.append(list(image_array))
        y_train.append(code['Pre'])
        
    for img in tqdm(os.listdir(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Original\Pro')):
        image = cv2.imread(os.path.join(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Original\Pro',img),1)
        image_array = cv2.resize(image , (s,s))
        X_train.append(list(image_array))
        y_train.append(code['Pro'])
        
    plt.figure(figsize=(20,20))
    for n , i in enumerate(list(np.random.randint(0,len(X_train),36))) : 
        plt.subplot(6,6,n+1)
        plt.imshow(X_train[i])   
        plt.axis('off')
        plt.title(getcode(y_train[i]))
        
    len(X_train)
    
    X_test = []
    y_test = []
    for img in tqdm(os.listdir(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Segmented\Benign')):
        image = cv2.imread(os.path.join(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Segmented\Benign',img),1)
        if image is not None:
            image_array = cv2.resize(image , (s,s))
            X_test.append(list(image_array))
            y_test.append(code['Benign'])
        
    for img in tqdm(os.listdir(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Segmented\Early')):
        image = cv2.imread(os.path.join(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Segmented\Early',img),1)
        if image is not None:
            image_array = cv2.resize(image , (s,s))
            X_test.append(list(image_array))
            y_test.append(code['Early'])
        
    for img in tqdm(os.listdir(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Segmented\Pre')):
        image = cv2.imread(os.path.join(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Segmented\Pre',img),1)
        if image is not None:
            image_array = cv2.resize(image ,(s,s))
            X_test.append(list(image_array))
            y_test.append(code['Pre'])
        
    for img in tqdm(os.listdir(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Segmented\Pro')):
        image = cv2.imread(os.path.join(r'C:\Users\M AJAY\OneDrive\Desktop\ALL\47.Acute_lymphoblastic_leukemia_Classification_Based_on_Convolutional_Neural_Network (1)\AcuteLymphoblasticLukemia1\Code\AcuteLymphoblasticLukemiaDetection\media\Segmented\Pro',img),1)
        if image is not None:
            image_array = cv2.resize(image ,(s,s))
            X_test.append(list(image_array))
            y_test.append(code['Pro'])
        
    plt.figure(figsize=(20,20))
    for n , i in enumerate(list(np.random.randint(0,len(X_test),36))):
        plt.subplot(6,6,n+1)
        plt.imshow(X_test[i])   
        plt.axis('off')
        plt.title(getcode(y_test[i]))
        
    da=[]
    for i,j in zip(X_train,y_train):
        da.append([i,j])
    
    import random
    random.shuffle(da)
    
    len(da)
    
    X=[]
    y=[]
    for img,label in da:
        X.append(img)
        y.append(label)
        
    X=np.array(X)
    y=np.array(y)
    
    # Normalize pixel values to 0-1 range
    X = X.astype('float32') / 255.0
    
    # Convert test data to numpy array and normalize
    X_test_array = np.array(X_test)
    y_test_array = np.array(y_test)
    X_test_array = X_test_array.astype('float32') / 255.0
    
    from sklearn.model_selection import train_test_split
    xtrain,xtest,ytrain,ytest = train_test_split(X,y,train_size=0.8,shuffle=True)
    print("xtrain shape:", xtrain.shape)
    print("xtest shape:", xtest.shape)
    print("X_test_array (Segmented) shape:", X_test_array.shape)
    print("xtrain min/max values:", xtrain.min(), "/", xtrain.max())
    
    # ============================================
    # QCResNet - Quality-enhanced Convolutional ResNet
    # Custom architecture built from scratch for ALL Classification
    # ============================================
    
    from keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, MaxPooling2D
    from keras.layers import GlobalAveragePooling2D, Dense, Dropout
    from keras.models import Model
    
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
    
    # Build the QCResNet model
    model1 = build_qcresnet(input_shape=(224, 224, 3), num_classes=4)
    
    print("QCResNet Model Architecture Built Successfully!")
    print("=" * 50)
    print("QCResNet - Quality-enhanced Convolutional ResNet")
    print("Custom architecture for ALL Classification")
    print("=" * 50)
    
    model1.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('Model Details are : ')
    print(model1.summary())
    
    history=model1.fit(xtrain,ytrain,batch_size=128,
                  verbose=1,
                  validation_data=(xtest,ytest),epochs=10)
    
    model1.save('Acute_Lymphoblastic_Lukemia_Model.h5')
    
    y_pred = model1.predict(xtest)
    print('Prediction Shape is {}'.format(y_pred.shape))
    
    import matplotlib.pyplot as plt
    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    
    # Get the final accuracy values and convert to percentage
    final_accuracy = round(float(accuracy[-1]) * 100, 2)
    final_val_accuracy = round(float(val_accuracy[-1]) * 100, 2)
    
    # Create accuracy plot
    plt.figure(figsize=(10, 6))
    plt.plot(accuracy, label='Training Accuracy')
    plt.plot(val_accuracy, label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy over Epochs")
    plt.legend(['train','test'],loc='lower right')
    plt.tight_layout()
    plt.savefig('media/accuracy_plot.png')
    plt.close()
    
    # Save results to JSON file for later retrieval
    import json
    from datetime import datetime
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    results = {
        'accuracy': final_accuracy,
        'val_accuracy': final_val_accuracy,
        'loss': round(float(loss[-1]), 4),
        'val_loss': round(float(val_loss[-1]), 4),
        'epochs': len(accuracy),
        'accuracy_list': [round(float(a) * 100, 2) for a in accuracy],
        'val_accuracy_list': [round(float(a) * 100, 2) for a in val_accuracy],
        'loss_list': [round(float(l), 4) for l in loss],
        'val_loss_list': [round(float(l), 4) for l in val_loss],
        'trained_on': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'model_type': 'QCResNet (Quality-enhanced Convolutional ResNet)',
        'total_params': model1.count_params(),
        'trainable_params': sum([tf.keras.backend.count_params(w) for w in model1.trainable_weights])
    }
    with open('media/training_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    return render(request,'users/training.html', results)

def prediction(request):
    import numpy as np
    from keras import utils as image
    from keras.models import load_model
    import tensorflow as tf
    import io
    from PIL import Image
    import cv2
    
    # Load the trained model
    model = load_model(
        'Acute_Lymphoblastic_Lukemia_Model.h5',
        custom_objects={'softmax_v2': tf.nn.softmax}
    )
    
    # Define the label mappings
    code = {"Benign": 0, "Early": 1, "Pre": 2, "Pro": 3}
    reverse_code = {v: k for k, v in code.items()}
    
    def get_class_label(prediction):
        class_idx = np.argmax(prediction)
        confidence = np.max(prediction) * 100
        return reverse_code[class_idx], confidence
    
    if request.method == 'POST' and request.FILES.get('image'):
        try:
            # Get the uploaded image
            img_file = request.FILES['image']
            
            # Convert to PIL Image
            img = Image.open(io.BytesIO(img_file.read()))
            
            # Convert PIL Image to numpy array
            img_array = np.array(img)
            
            # If grayscale, convert to BGR (OpenCV format used in training)
            if len(img_array.shape) == 2:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
            elif img_array.shape[2] == 4:  # RGBA
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
            else:  # RGB from PIL - convert to BGR to match training
                img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Resize to match training size (224x224)
            img_resized = cv2.resize(img_array, (224, 224))
            
            # Normalize pixel values to 0-1 range (same as training)
            img_normalized = img_resized.astype('float32') / 255.0
            
            # Add batch dimension
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            # Make a prediction
            predictions = model.predict(img_batch, verbose=0)
            predicted_class, confidence = get_class_label(predictions[0])
            
            # Get all probabilities for debugging
            all_probs = {reverse_code[i]: float(predictions[0][i]) * 100 for i in range(4)}
            
            return render(request, 'users/prediction.html', {
                'predicted_class': predicted_class,
                'confidence': round(confidence, 2),
                'all_probabilities': all_probs
            })
        except Exception as e:
            return render(request, 'users/prediction.html', {
                'error': f'Prediction failed: {str(e)}'
            })
    
    return render(request, 'users/prediction.html')