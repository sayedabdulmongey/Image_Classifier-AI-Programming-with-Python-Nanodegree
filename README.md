# Image Classifier Project

An image classifier project involves creating an artificial intelligence model capable of categorizing images into predefined classes. This technology is widely used across diverse applications such as facial recognition, medical image analysis, and content filtering. 

## Key Components

- **Dataset:** A well-labeled dataset is crucial for training a robust image classifier.
  
- **Preprocessing:** Images often need preprocessing before being fed into the model. Common steps include resizing, normalization, and augmentation.

- **Model Architecture:** Convolutional Neural Networks (CNNs) are commonly used for their ability to capture spatial hierarchies and patterns in images.

- **Training:** The model learns to associate patterns and features with specific classes through iterative adjustments of parameters.

- **Validation:** The model is evaluated on a separate dataset to ensure generalization to new, unseen data.

- **Hyperparameter Tuning:** Fine-tuning model hyperparameters can significantly impact performance.

- **Testing:** The final model is tested on an independent dataset to assess accuracy and performance.

- **Deployment:** Once satisfied with performance, the model can be deployed for real-world use.

Image classifier projects are prevalent in various domains, contributing to advancements in healthcare, security, and computer vision. With advancements in deep learning and powerful frameworks like TensorFlow and PyTorch, building and deploying image classifiers has become more accessible for developers and researchers.

# Flower Recognition with Pre-trained Models

## Introduction

The Flower Recognition project is designed to create an image classifier capable of identifying various flower species from a dataset containing 102 flower categories. This project utilizes the power of pre-trained models, specifically VGG16 and VGG13, to streamline the development process by leveraging knowledge gained from extensive training on large-scale image datasets.

# Command Line Application
* Train a new network on a data set with ```train.py```
  * Basic Usage : ```python train.py data_directory```
  * Prints out current epoch, training loss, validation loss, and validation accuracy as the netowrk trains
  * Options:
    * Set direcotry to save checkpoints: ```python train.py data_dor --save_dir save_directory```
    * Choose arcitecture (vgg13 or vgg16 available): ```pytnon train.py data_dir --arch "vgg16"```
    * Set hyperparameters: ```python train.py data_dir --learning_rate 0.001 --hidden_layer1 120 --epochs 20 ```
    * Use GPU for training: ```python train.py data_dir --gpu gpu```
    
* Predict flower name from an image with ```predict.py``` along with the probability of that name. That is you'll pass in a single image ```/path/to/image``` and return the flower name and class probability
  * Basic usage: ```python predict.py /path/to/image checkpoint```
  * Options:
    * Return top **K** most likely classes:``` python predict.py input checkpoint ---top_k 3```
    * Use a mapping of categories to real names: ```python predict.py input checkpoint --category_names cat_To_name.json```
    * Use GPU for inference: ```python predict.py input checkpoint --gpu```

# Hyperparameters
As you can see you have a wide selection of hyperparameters available and you can get even more by making small modifications to the code. Thus it may seem overly complicated to choose the right ones especially if the training needs at least 15 minutes to be completed. So here are some hints:
* By increasing the number of epochs the accuracy of the network on the training set gets better and better however be careful because if you pick a large number of epochs the network won't generalize well, that is to say it will have high accuracy on the training image and low accuracy on the test images. Eg: training for 5 epochs training accuracy: 88% Test accuracy: 56%. 
* A big learning rate guarantees that the network will converge fast to a small error but it will constantly overshot
* A small learning rate guarantees that the network will reach greater accuracies but the learning process will take longer


## Pre-Trained Network
The checkpoint.pth file contains the information of a network trained to recognise 102 different species of flowers. 
In order to have a prediction for an image located in the path /path/to/image using my pretrained model you can simply type ```python predict.py /path/to/image checkpoint.pth```.


## Finally 
### [Certification](https://graduation.udacity.com/confirm/e/7a46d530-6c23-11ee-a38b-d30991bd2d11)
