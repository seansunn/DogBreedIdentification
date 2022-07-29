import cv2
import numpy as np
import tensorflow as tf
from keras.utils import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input


# load necessary files
dog_names = list(np.load('data/dog_names.npy'))
face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')
ResNet50_model = ResNet50(weights='imagenet')
classify_model = tf.keras.models.load_model('data/dogmodel_output.h5')


# dog detector
def dog_detector(img_path):
    img = preprocess_input(path_to_tensor(img_path))
    prediction = np.argmax(ResNet50_model.predict(img, verbose=0))
    return ((prediction <= 268) & (prediction >= 151))


# Human face detector
def face_detector(img_path):
    # loads image from path
    img = cv2.imread(img_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    return len(faces) > 0


# turns image from path to tensor for prediction
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image
    img = load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


# use ResNet50 model to predict breed
def predict_breed(img_path):
    prediction = classify_model.predict(path_to_tensor(img_path), verbose=0)
    return dog_names[np.argmax(prediction)]


# clean the output string
def clean(result):
    return ' '.join([i.capitalize() for i in result[1:].split('_')])


# main
def classifier(img_path):
    
    # detect if dog
    P_dog = dog_detector(img_path)
    
    # if a dog is detected (probability >= 0.8) in the image, return the predicted breed
    if P_dog >= 0.8:
        result = predict_breed(img_path)
        return f'A dog is detected! It is a...{clean(result)}!'
    
    # if a dog is not detected (probability < 0.8), detect if a human face is presented
    else:
        # detect if face
        P_face = face_detector(img_path)
        
        # if a human is detected in the image, return the resembling dog breed.
        if P_face >= 0.8:
            result = predict_breed(img_path)
            return f'A human is detected! The resembling dog is...{clean(result)}!'
        
        # if neither is detected in the image, return the result
        else:
            return 'No dogs or humans detected!'