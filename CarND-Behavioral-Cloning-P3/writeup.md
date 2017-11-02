# **Behavioral Cloning** 

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[data1]: ./training_data_raw.JPG
[data2]: ./training_data_absolute.JPG
[data3]: ./training_data_raw_with_augmentation.JPG
[data4]: ./training_data_absolute_with_augmentation.JPG


## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

###  Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run1.mp4 video of car driving in autonomous mode

### Submission includes functional code
The drive.py was modified to set the speed to 30. Using the Udacity simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

The model is the one from  Nvidia's End to End Learning for Self-Driving Cars paper with relu activations.  

####2. Attempts to reduce overfitting in the model

The training was terminated after 5 epochs to reduce overfitting.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

####1. Solution Design Approach

First, I started off with the LeNet model and data generated from driving around the track in both directions. The car mostly stayed on the track, but would always leave the track at the turn after the bridge. 

Second, I tried to augment the data using the left and right images and various steering correct factors. This made model perform horribly. I tried to modify the model in various ways by adding more convolutional layers, changing the out of the convolution sizes, and adding dropout. Nothing I did seemed to improve from the first approach. Therefore, I stop trying to use the left and right images. Later, I would realize that I had the signs on the steering correction factors backwards.

Third, to try to get the car to not leave the track on the turn after the bridge, I generated data of the car successfully handling curves. Then, I used a higher proportion of this data along with the original data to train the network. Again, I tried various modification to Lenet without getting any good results.

Fourth, I took all the training data and flipped each images so that I had equal numbers of images turning left and right. This would actually improve performance of whatever network I was using. This got me on the path to generating the training data that I detail below. Once I generated the new training data, I was able to get the car to stay on the road with almost every variation of the LeNet model that I tried.

####2. Final Model Architecture

I ended up using the model from Nvidia's End to End Learning for Self-Driving Cars paper. I used relu activations for all layers except the last. Since, the network did not show maxpooling, the convolutions use a stride of 2. Although, I was able to get various version of the LeNet network to also work, Nvidia's network seemed to provide the smoothest and most lane centered drive.

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
