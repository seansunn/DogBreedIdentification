# Dog Classifier Web App

## Project descripton

This Flask web app is able to classify dog's breed from a dog image, or a human's resembling dog if the image contains a human face. The accuracy of prediction is ~83%. The pre-trained deep neural network model for the classification task is based on ResNet50 with weights from imagent, and the output layers are created and tuned to suit the dog classification task.

## File description

data\
|- dog_names.npy `numpy exported file contains a list of dog breeds names`\
|- haarcascade_frontalface_alt.xml `cv2 frontal face recognition file`\
|- dogmodel_output.h5 `pre-trained classification model based on ResNet50`\
templates\
|- index.html `main page of web app`\
app.py `flask file that runs app`\
model.py `model file that contains necessary functions for classification task`\
requirements.txt `list of required packages for running the web app`\
README.md

## Instructions

1. Create and activate a virtual environment and follow requirements.txt to install all the required packages for app.

2. Clone the project, create an empty folder named `uploads` in the project's directory, and run the following command to run web app:
    `python app.py`

3. The app should be running on `http://127.0.0.1:5000/`

4. In the web app, select a picture from your computer and submit to see the classification result.

## Licensing, Acknowledgements
Feel free to use any of the code.
