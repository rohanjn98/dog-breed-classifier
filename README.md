# Dog Breed Classifier

This is the capstone project of my MLE (Machine Learning Engineer) Nanodegree. 

## Overview
Multi-class classification using CNN (Convolutional Neural Network)
is an interesting part of the ML domain. In this, we develop a model
that is capable of predicting a class to which the input belongs. In our
case, the input is an image. The problem at hand is to classify the
input image into a dog breed class. If the input image is of a dog then
the model should correctly classify it to a breed whereas if the input
image is not of a dog, the model should predict the most resembling
breed for the input image. We choose CNN as the way to go because
this problem statement involves dealing with images, processing
them, etc.

## Problem Statement
We need to develop a model (deep neural network) to classify the
input into different classes. The input may or may not be a dog
image. Depending on the input image, the model should predict the
class. If the input image is of a dog, the correct breed of that dog
must be predicted. If the input image is not of a dog (let’s say it is that
of a human being) then the model must predict the most resembling
breed that the input showcases. Once the model is developed, it should be tested for a minimum of 6 images supplied by the user.
(This is showcased is the last cell output in the dog_app.ipynb)

## Exploratory Data Analysis
1. The dog image dataset can be downloaded from here: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/dogImages.zip
2. The human image dataset can be downloaded from here: https://s3-us-west-1.amazonaws.com/udacity-aind/dog-project/lfw.zip
3. Total Dog Images: 8351; Total Human Images: 13233

## Data Preprocessing
Data pre-processing is necessary for every ML problem. The need for data pre-processing arises due to the imbalanced dataset and to make sure that input images have same dimensions.
Transforms are used to resize the images, ex: RandomResizeCrop (224), RandomHorizontalFlip and RandomRotation (15). The
standard input image size for popular models like VGG-16 is of 224X224 generally. Using a similar size would be preferred.
Normalization is applied to all the 3 datasets.
1. Train data (train_dataset) - Augmentation is done using the above
mentioned transforms to avoid overfitting.
2. Validation data (val_dataset) - Image Resizing - 224X224
3. Test data (test_dataset) - Image Resizing - 224X224

## Algorithm / Modeling
I have solved the problem statements by using these 4 steps:
1. Detecting Human Faces: The first step is to detect if a human being is present in the input image or not. This is achieved by
using OpenCV's implementation of Haar feature-based cascade classifiers to detect human faces in images.
The human face detector runs very efficiently and the results are as follows: Percentage Classified as Human Faces: 98 - From Human
folder; Percentage Misclassified as Human Faces: 17 - From dog folder. These results are also available in the dog_app.ipynb
2. Detecting Dogs: The first step is to detect if the image input is that of a dog or not. This is accomplished using a pre-trained
VGG-16 model which was imported into the code using the torchvision.models.
Percentage Classified as Dog Faces: 100 - from dog folder Percentage Misclassified as Dog Faces: 1 - from human folder
These results are also available in the dog_app.ipynb
3. Creating a CNN from scratch without transfer learning: Activation Function = ReLU - This is decided based on our
application at hand. In order to avoid overfitting of the model, a dropout of p=0.25 is added. Finally, we have an output tensor of
113 which equals the breed categories - total classes available for this multi-class classification problem. I have used the
cross-entropy loss as the loss function and SGD optimizer and learning rate alpha=0.02. This model after running for 20
epochs has produced a 16% test accuracy (141/836) which is better than the expected 10%.
4. Creating a CNN from scratch using Transfer Learning: I have used a deep neural net to achieve this task.
Since we need to use transfer learning, I decided to consider a pre-trained model that is already trained on a large image
dataset like the ImageNet dataset. So I chose the Resnet101 which is already present in the torchvision.models. I imported
the model from there. I managed to achieve a test accuracy of 83% (701/836) which is evident from the cell outputs after
training the model for 20 epochs. This accuracy is much higher than the required 60%. The fully connected layer is changed to
133 for the problem at hand since we have 133 breed classes. CrossEntropyLoss is chosen as the loss function with a learning
rate of 0.02. These are standard settings for multi-class classification problems.

## Benchmark & Metrics
The loss function selected is the cross entropy loss.<br>
Accuracy = Correctly classified items / All classified items<br>
The expected accuracies are as follows:
- Scratch CNN without transfer learning = 10%, Result = 16%
- Scratch CNN with transfer learning = 60%, Result = 83% using Resnet101 architecture

## Results
<p float="left">
  <img src="/Output1.png" width="300" height="470"/>
  <img src="/Output2.png" width="300" height="470"/> 
  <img src="/Output3.png" width="300" height="470"/>
</p>

## Conclusion
The model that I have developed works well and surpasses the expected accuracy. This solves our problem
statement of correctly classifying an input image to a dog breed class. The achieved accuracy for the model is 83% test accuracy that far
exceeds the expected 60%. There are a few things that might help improve the model performance that I have also enlisted in the
dog_app.ipynb. Some of them include:
- The model could be trained for higher epochs (>20) that might
improve the current metrics.
- Fine-tuning of parameters could be done on the basis of train, valid
and test loss curves or metrics.
- Since this can be deployed on a mobile or web app, A/B testing could
be done with different sets of parameters at once and then decide on
what parameter values to be chosen. This is important as we need to
stay relevant with input data and the data available for training.
- Last but not the least, more image augmentation techniques could be
used to virtually increase the dataset size, this might be helpful for
better model performance.
