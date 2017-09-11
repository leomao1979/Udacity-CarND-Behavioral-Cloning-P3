# Behaviorial Cloning Project

[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Overview
---
In this project, you will use what you've learned about deep neural networks and convolutional neural networks to clone driving behavior. You will train, validate and test a model using Keras. The model will output a steering angle to an autonomous vehicle.

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[nvidia_architecture]: ./nvidia_architecture.png "Nvidia's Network Architecture"
[center_driving]: ./examples/center_driving.jpg "Center Driving"
[flipped_center_driving]: ./examples/flipped_center_driving.jpg "Flipped Center Driving"
[recover_right]: ./examples/recover_right.jpg "Recover Right"
[recover_right_center]: ./examples/recover_right_center.jpg "Recover Right Center"
[recover_center]: ./examples/recover_center.jpg "Recover Center"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I tried different models, from LeNet-5, Nvidia's network to pre-trained VGG16. Finally the Nvidia's network has been employed.

The model consists of 5 convolutional layers. The first three layers use strided convolutions with 5x5 kernel and 2x2 strides, and the final two use non-strided convolutions with 3x3 kernel. The depths are between 24 and 64 (model.py lines 104-108).

The model includes RELU layers to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 103).

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 111 and 113).
It was trained and validated on different data sets to avoid overfitting (model.py lines 139-152).
The model was tested by running through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model uses an adam optimizer, so the learning rate was not tuned manually (model.py line 148).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving counter-clockwise.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use the classical LeNet-5 model (model.py lines 84-98). It started to drive, though looks like a drunk driver, after trained with the center images of Udacity sample data only. Then I augmented the data by using images from left and right cameras, and flipping horizontally. After trained with augmented data the model performed better, but still failed at the left turn right after driving through the bridge.

Then I turned to Nvidia's End-to-End deep learning solution for self driving car (https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/). In the first trial run the vehicle would go off track. It didn't recover back to center when it veered off to sides, so I increased the steering angle adjustment for side camera images, collected more recovery driving data and more turning data. Then I implemented the generator to make the training process memory efficient. Meanwhile, I also added dropout to the fully connected layers to combat overfitting. After all these tunings, the trained model could drive the vehicle autonomously around track one without leaving the road.

Next I tried VGG16 model with weights pre-trained on ImageNet (model.py lines 118-136). I removed the top layers of the network and added my fully connected layers. All remaining VGG16 layers are set to non-trainable, so we will train the weights of new layers only. Comparing to LeNet and Nvidia models, the training process takes longer and requires more memory. And, it didn't demonstrate any improvement when test in the simulator. So, the Nvidia network is finally employed.

#### 2. Final Model Architecture

The final model architecture (model.py lines 100-116) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 160x320x3 RGB image   							              |
| Cropping2D        | cropping ((70, 24), (0, 0)), outputs 66x320x3     |
| Lambda            | resize to 66x200 and normalize, outputs 66x200x3  |
| Convolution 5x5   | 2x2 stride, valid padding, outputs 31x98x24, RELU |
| Convolution 5x5   | 2x2 stride, valid padding, outputs 14x47x36, RELU |
| Convolution 5x5   | 2x2 stride, valid padding, outputs 5x22x48, RELU  |
| Convolution 3x3	  | 1x1 stride, valid padding, outputs 3x20x64, RELU  |
| Convolution 3x3	  | 1x1 stride, valid padding, outputs 1x18x64, RELU  |
| Flatten	      	  | outputs 1152				                              |
| Fully connected		| 1152x100 weights, outputs 100, Dropout  					|
| Fully connected		| 100x50 weights, outputs 50, Dropout  							|
| Fully connected		| 50x10 weights, outputs 10       									|
| Fully connected		| 10x1 weights, outputs 1       									  |

Here is a visualization of the architecture

![Nvidia Network Architecture][nvidia_architecture]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![Center Lane Driving][center_driving]

I then recorded the vehicle recovering from the left and right sides of the road back to center. These images show what a recovery looks like:

![Recover Right][recover_right]

![Recover Right Center][recover_right_center]

![Recover Center][recover_center]

To augment the data set, I also flipped images and angles to reduce the left turn bias from the nature of track one. For example, here is an image that has been flipped:

Image captured by center camera:

![Center Driving][center_driving]

Flipped image:

![Flipped Center Driving][flipped_center_driving]

I randomly shuffled the data set and put 20% of the data into a validation set (model.py line 140).

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by the mean squared error on validation set.
