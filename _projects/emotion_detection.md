---
layout: emotion_detection
title: Emotion Detection
description: a project for my application for DOST
img: assets/img/3.jpg
importance: 2
category: personal
---


I created a simple tool that takes in a picture of a person as an input then tries to predict the facial expression that person is showing. I used Google Colab to train the model and deployed it to a simple flask app with the help of pythonanywhere. 

The model was not very accurate when I tested it using different pictures of myself. Improving the dataset and playing around with the neural network can improve the accuracy of the model. This project focused more on learning how to execute the whole machine learning workflow. Starting from training a machine learning model to deploying it as an API. 

(picture)
### Code and Demo

The code for the training and deployment can be found at this  [github repository](https://github.com/studentmorrisjohn/emotion_detection). A simple demo of the tool can be found at the bottom of this page.

### Data Set

I've used a Convolutional Neural Network, or CNN,  to recognize seven types of expression: angry, disgust, fear, happy, sad, surprise, neutral. The model will be trained on the [FER 2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013) The data consists of 48x48 pixel grayscale images of faces. The training set consists of 28,709 examples and the public test set consists of 3,589 examples.

### Pipeline

I trained the model on V100 GPU on Google Colab Pro using keras and served the resulting model as an API by creating a simple Flask APP deployed on pythonanywhere. 

(picture)

### Training on Google Colab Pro

Google Colab, is a free, cloud-based platform provided by Google that allows users to write and execute Python code in a collaborative environment. It is built on top of Jupyter Notebooks and provides a similar interface, making it easy for users to write and execute code in a step-by-step manner.

I subscribed to Colab pro to have access to larger RAM and faster GPU runtimes. But this project can also be done on the free version of Colab. 

#### Imports

For this project, I've used keras, tensorflow, opencv, and numpy for training the data. Matplotlib was used for some visualization but was not used in the actual training of the model.
```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
```

#### Loading the Data

I've gotten the dataset from kaggle and downloaded it as a zip file. I then uploaded that zip file to Google Colab. After uploading I unzipped the file and gotten two folders for the training set and the test set.
```
!unzip /content/fer2013.zip
```

The dataset is separated into to folders, a training set and a test set. This made the splitting of the data for training and validation easier. I just loaded the images from the training folder using opencv and put it in a list called training data. Then extracted the x and y. I implemented the same way of loading for the test data.

***Setting the variables
```python
# directory variable
dataDirectory = "train/"
dataDirectoryTest = "test/"

# expression categories
categories = ["angry","disgust","fear","happy","neutral","sad","surprise"]

# image size the images will be resized to for the processing
img_size = 224

# initialize variables
trainingData = []
testData = []
```

***Loading the training data***
```python
for category in categories:
  path = os.path.join(dataDirectory, category)
  class_num = categories.index(category)
  count = 0

  for img in os.listdir(path):
    try:
      if count == 436:
        break
        
      img_array = cv2.imread(os.path.join(path,img))
      new_array= cv2.resize(img_array,(img_size,img_size))
      trainingData.append([new_array,class_num])
      
      count += 1

    except Exception as e:
      pass
```

For the loading of the test data, I've put a limit of 436 for the total maximum number of data per classification. The reason behind this is that the amount of pictures per classification was not the same. I used 436 because this is the size of the classification, disgust, with the fewest data. I did not impose this limit on the test data since it will not affect the training of the model.

***Loading the test data***
```python
# looping through each category in the directory
for category in categories:
  path = os.path.join(dataDirectoryTest, category)
  class_num = categories.index(category)
  
  for img in os.listdir(path):
    try:
      img_array = cv2.imread(os.path.join(path,img))
      new_array= cv2.resize(img_array,(img_size,img_size))
      testData.append([new_array,class_num])

    except Exception as e:
      pass
```

***Extracting the features and label from the data sets***
```python
# instantiating the x and y variables
x = []
y = []

x_test = []
y_test = []

# extracting the features and label from the training data and test data
for features, label in trainingData:
    x.append(features)
    y.append(label)

for features, label in testData:
    x_test.append(features)
    y_test.append(label)
```

#### Preprocessing the Data

I reshaped the data in the format accepted by the model we will be using
The model will take batches of the shape `[N, 224, 224, 3]` and outputs probabilities of the shape `[N, 7]`.

```python
# reshaping and normalization
x = np.array(x).reshape(-1, img_size, img_size, 3)
x_test = np.array(x_test).reshape(-1, img_size, img_size, 3)

x=x/255.0
x_test = x_test/255.0

# converting the list to np array
y = np.array(y)
y_test = np.array(y_test)
``` 

#### Creating the model

I created a simple convolutional neural network based off of MobileNet. A CNN model which is known for their efficiency and suitability for deployment on devices with limited computational resources. The following model contains 2 activation layers and 3 dense layers.

```python
# base model
model = tf.keras.applications.MobileNetV2()
base_input = model.layers[0].input
base_output = model.layers[-2].output

# layers to modify the base model
final_output = layers.Dense(128)(base_output)
final_ouput = layers.Activation('relu')(final_output)
final_output = layers.Dense(64)(final_ouput)
final_ouput = layers.Activation('relu')(final_output)
final_output = layers.Dense(7, activation='softmax')(final_ouput)

# creating a new model out of the old model using the layers to modify the model
new_model = keras.Model(inputs = base_input, outputs = final_output)
new_model.compile(loss="sparse_categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"] )
```

#### Fit and Validate

After defining the model, I trained the model for 25 epochs. And validated it using the test data

```python
# training the model
new_model.fit(x,y, epochs = 25)

# evaluating the model
score = new_model.evaluate(x_test, y_test, verbose=0)
```

#### Saving the model for the API

After the training and validation of the model, I saved the model into an h5 file. I will use this file to make predictions using my API.

```python
new_model.save("emotiondetection.h5")
```


### Deploying using Flask and PythonAnywhere

I created a simple API endpoint where I can send a picture and the flask app will execute the code to make a prediction. Flask is a lightweight and web framework for Python that is designed to be simple and easy to use. It is widely used for developing web applications, APIs (Application Programming Interfaces), and other web-related projects. 

To host this flask app and be accessible through the internet, I used pythonanywhere. PythonAnywhere is an online platform that provides a Python development environment in the cloud. It allows users to write, run, and host Python applications without the need for local installations or server management.

#### Imports

For the simple flask api, I used flask, pillow, tensorflow, opencv, and numpy

```python
from flask import Flask, request, render_template, jsonify
from PIL import Image
from io import BytesIO

import tensorflow as tf
import cv2
import os
import numpy as np
```
#### Loading the Model

To load the model into the app, I used karas `load_model` method.

```python
new_model = tf.keras.models.load_model("/path/to/emotiondetection.h5")
```

#### Processing the image

Before making a prediction using the model, I needed to do three things. First is to turn the whole picture into a grayscale image. Next is to detect the face of the person in the picture. For this, I have used open cv's cascade classifier. Lastly is to preprocess the picture of the face to match the format needed by the model. To do this, I zoomed in on the detected face and resized the picture. 

***Turning into grayscale***
```python
# Converting image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

***Face Detection***
```python
# Reading the image
img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

# Loading the required haar-cascade xml classifier file
haar_cascade = cv2.CascadeClassifier('/path/to/haarcascade_frontalface_default.xml')

# Applying the face detection method on the grayscale image

faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)
```

***Zooming and Resizing***
```python
# Zooming in on face
for x,y,w,h in faces_rect:
	roi_gray = gray_img[y:y+h, x:x+w]
	roi_color = img[y:y+h, x:x+w]
	cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
	facess = haar_cascade.detectMultiScale(roi_gray)
	
	if len(facess) == 0:
		print("Face not detected")
	else:
		for(ex,ey,ew,eh) in facess:
			face_roi = roi_color[ey: ey+eh, ex: ex+ew]

# Resizing image before making a prediction
final_image = cv2.resize(face_roi, (224,224))
final_image = np.expand_dims(final_image, axis=0)
final_image = final_image/255.0
```

#### Making a prediction

After the preprocessing stage, I used the `predict` method built in to the loaded model. This will return a numpy array of the different scores for the classifications. I used argmax to determine which of the classification has the highest score. Then I use this as an index to get the name of  the classification in a classifications list.

```python
classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]

prediction = classes[np.argmax(Predictions)]
```


![[Pasted image 20231114143633.png]]
*Result of the training*
#### Creating the `make_predicton` method

I then put the loading of the model, processing of the image, and the making of prediction in to a function called `make_prediction`.

```python
def make_prediction(image):
    # Load the model
    new_model = tf.keras.models.load_model("/path/to/emotiondetection.h5")

    classes = ["angry","disgust","fear","happy","neutral","sad","surprise"]
    
    # Reading the image
    img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # Converting image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Loading the required haar-cascade xml classifier file
    haar_cascade = cv2.CascadeClassifier('/path/to/haarcascade_frontalface_default.xml')

    # Applying the face detection method on the grayscale image
    faces_rect = haar_cascade.detectMultiScale(gray_img, 1.1, 9)

    # Zooming in on face
    for x,y,w,h in faces_rect:
        roi_gray = gray_img[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
        
        facess = haar_cascade.detectMultiScale(roi_gray)
        
        if len(facess) == 0:
            print("Face not detected")
        else:
            for(ex,ey,ew,eh) in facess:
                face_roi = roi_color[ey: ey+eh, ex: ex+ew]

    # Resizing image before making a prediction
    final_image = cv2.resize(face_roi, (224,224))
    final_image = np.expand_dims(final_image, axis=0)
    final_image = final_image/255.0
    
    # Making the Prediction
    predictions = new_model.predict(final_image)

    prediction = classes[np.argmax(predictions)]

    return  prediction
```

#### Creating the API Endpoint

I created a route that will accept a `POST` request with an image in its body. It will then process the said image and use the `make_prediction` method to create a prediction and return it as a JSON.

```python
@app.route('/process_image', methods=['POST'])
def process_image():
    # Check if the POST request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    
    # Check if the file is empty
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    try:
        # Read the image file
        img = Image.open(file)
        image, prediction = make_prediction(img)

        # Process the image (you can replace this with your own logic)
        result_string = prediction
        return jsonify({'result': result_string})

    except Exception as e:
        return jsonify({'error': str(e)})
```

#### Deploying to pythonanywhere

The deployment of this API to pythonanywhere is simple. I just created an account, uploaded the files to the dedicated file section. And follow the instructions detailed in their documentation on deployment of a flask app.

### Conclusion

This project has thought me the process of training a machine learning model and deploying it to an API. I have little experience with working on machine learning so the accuracy of the model I have trained is pretty low. You can test it in a demo below, but don't expect it to blow your mind. This project serves only as a proof of concept. I plan on improving the accuracy and developing a user interface for an app some time later. 

### Demo

To try this machine learning model out, upload a photo of a person. Make sure that the person is facing the camera and their expression is visible. Don't upload a picture with multiple people in it.

