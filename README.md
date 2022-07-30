# Dog Classifier

## Contents
[0. File description](https://github.com/seansunn/DogClassifier/blob/main/README.md#file-description)\
[1. Web app instructions](https://github.com/seansunn/DogClassifier/blob/main/README.md#web-app-instructions)\
[2. Project Introduction](https://github.com/seansunn/DogClassifier/blob/main/README.md#project-introduction)\
[3. Strategy](https://github.com/seansunn/DogClassifier/blob/main/README.md#metrics)\
[4. EDA](https://github.com/seansunn/DogClassifier/blob/main/README.md#eda)\
[5. Modelling](https://github.com/seansunn/DogClassifier/blob/main/README.md#odelling)\
[6. Hyperparameter tuning](https://github.com/seansunn/DogClassifier/blob/main/README.md#hyperparameter-tuning)\
[7. Results and conclusion](https://github.com/seansunn/DogClassifier/blob/main/README.md#results-and-conclusion)\
[8. Improvements](https://github.com/seansunn/DogClassifier/blob/main/README.md#improvements)\
[9. Licensing and acknowledgements](https://github.com/seansunn/DogClassifier/blob/main/README.md#licensing-and-acknowledgements)


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


## Web app instructions
1. Create and activate a virtual environment and follow `requirements.txt` to install all the required packages for app;
2. Clone the project, create an empty folder named `uploads` in the project's directory, and run the following command to run web app:
    `python app.py`
3. The app should be running on `http://127.0.0.1:5000/`;
4. In the web app, select a picture from your computer and submit to see the classification result.


## Project Introduction
This project aims to build a pipeline to process and classify real-world, user-supplied images. If provided an image of a dog, the web app will identify an estimate of the dog's breed; if supplied an image of a human, the code will identify the resembling dog breed.


## Strategy
In the pipeline, firstly, ResNet50 (with weights from imagenet) and opencv are used to detect if the uploaded image is a dog or a human face respectively. If detected, then a convolution neural network built through transfer learning (base model: ResNet50) will be used to classify the image and return the result. Finally, the pipeline is deployed as a web app by using flask.

The dataset for training the convolution neural network is provided by Udacity:
1. dog dataset: [link here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip)
2. human dataset: [link here](https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip)

The metrics used to evaluate the performance of the pipeline is `accuracy`.


## EDA
The images are loaded by scikit learn and numpy, and splitted into training, validation, and test sets. In total, there are `133` dog categories and `8351` total dog images: `6680` images for training set, `835` images for validation set, and `836` for test set.


## Modelling
The convolution neural network is built through transfer learning on top of ResNet50. In the CNN architecture:
1. used `GlobalAveragePooling2D layer` to shrink the size of the input and ease the computations for parameters;
2. used `Dropout layer` that sets the randomly choosed input units to 0, which helps prevent overfitting;
3. used a `Dense layer` (fully-connected layer) and softmax as activation to calculate the probability of each label, and pick the one with the highest probability.

The model is compiled with `Adam` optimizer, and the loss used is `categorical crossentropy`.


## Hyperparameter tuning
The tuned hyperparameters are:\
`learning rate for Adam` = .0003 (tuned in logarithmic scale, e.g. .01, .001, .0001...)\
`Dropout probability` = .4 (based on validation accuracy)\
`epochs` = 20 (based on how many loss is reduced on each epoch)\
`batch size` = 64 (based on validation accuracy)


## Results and Conclusion
The accuracy of the model's prediction on the test set is ~83%. The pipeline is able to identify if the uploaded image is a dog or a human face or neither, and then it is able to predict the dog's breed or resembling dog's breed according to the uploaded image.


## Improvements
The training accuracy achieved is `~0.98`, but the validation accuracy is `0.84` and the test accuracy is `0.83`. This indicates that the variance can be improved.
In order to reduce the variance:
1. get more dog images for training, or use data augmentation to generate more images;
2. adjust and tune the regularization to prevent overfitting;
3. may use a different architecture for the task, such as the inception network.


## Licensing, Acknowledgements
Thanks to Udacity for providing the dataset. Feel free to use any of the code.
