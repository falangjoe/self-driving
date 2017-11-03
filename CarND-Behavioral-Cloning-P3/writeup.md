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

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
## Files Submitted & Code Quality

### Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup.md summarizing the results
* run1.mp4 video of car driving in autonomous mode

### Submission includes functional code
The drive.py was modified to a speed of 30. Using the Udacity simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

### Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

### 1. An appropriate model architecture has been employed

The model is the one from  Nvidia's End to End Learning for Self-Driving Cars paper with relu activations.  

### 2. Attempts to reduce overfitting in the model

The training was terminated after 5 epochs to reduce overfitting. Further epochs would not have reduced the validation loss.

### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

### 4. Appropriate training data

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

### 1. Solution Design Approach

First, I started off with the LeNet model and data generated from driving around the track in both directions. The car mostly stayed on the track, but would always leave the track at the turn after the bridge. 

Second, I tried to augment the data using the left and right images and various steering correction factors. This made the model perform horribly. I tried to modify the model in various ways by adding more convolutional layers, changing out of the convolution sizes, and adding dropout. Nothing that I did seemed to improve from the first approach. Therefore, I stop trying to use the left and right images. Later, I would realize that I had the signs on the steering correction factors backwards.

Third, to try to get the car to not leave the track on the turn after the bridge, I generated data of the car successfully handling curves. Then, I used a higher proportion of this data along with the original data to train the network. Again, I tried various modification to LeNet without getting any good results.

Fourth, I took all the training data and flipped each center image so that I had equal numbers of images turning left and right. This actually improved the performance of whatever network I was using. This got me on the path to generating the training data that I detail below. Once I generated the new training data, I was able to get the car to stay on the road with almost every variation of the LeNet model that I tried.

### 2. Final Model Architecture

I ended up using the model from Nvidia's End to End Learning for Self-Driving Cars paper. I used relu activations for all layers except the last. Since, the network did not show maxpooling, the convolutions use a stride of 2. Although, I was able to get various version of the LeNet network to also work, Nvidia's network seemed to provide the smoothest and most lane centered drive.

### 3. Creation of the Training Set & Training Process

After realizing that using equal numbers of left and right steering angle images improved network performance. I started to look at the problem differently. The simulator only outputs a finite number of steering angles. Therefore, you could look at the network as a classification problem taking an image of a good driver driving in a lane and outputing the probabilities of what steering angle they should be using. This got me to think that just like training a classifier, you would want to have relatively the same number of training examples of each steering angle (i.e class). Below is a graph of the data collected by driving the simulator with the steering angles in degrees.

![alt text][data1]

You can see that a large number of driving angles are clustered between (-2.5,2.5) degrees. There were actually around 10,000 samples at 0 degress, but I truncated the graph at 2000 examples. Hence training on this data should give the network a bias to returning small angles to minimize the loss, and not performing well on curves that need larger angles.

Thus, I grouped frames by the absolute of their steering angle and picked a random sample of 200 training images at each angle. Then, I randomly flipped the images to get equal numbers of left and right steering angles. Training on this data made the network perform a lot better, and the car would almost make it around the corner after the bridge.

Since, I now had something working pretty good, I went back to trying to add the left and right images to the training pipeline. I actually examined the images and realized I was using the wrong sign on the correction factor. I then added these extra images with the steering angle corrected to my pipeline and choose 600 images of each angle. This finally got the car to stay on the track in autonomous mode.

Once everything was working I switched to Nvidia's network and played around with the steering correction factor. I settled on a correction factor of 0.15 = 0.15 * 25 = 2.75 degrees. This seemed to provide the smoothest driving. 
