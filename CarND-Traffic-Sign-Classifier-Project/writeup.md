# **Traffic Sign Recognition** 

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./training_data_barchart.JPG "Training Data Visualization"
[prediction0]: ./predictions/0.JPG
[prediction1]: ./predictions/1.JPG
[prediction2]: ./predictions/2.JPG
[prediction8]: ./predictions/8.JPG
[prediction11]: ./predictions/11.JPG
[prediction12]: ./predictions/12.JPG
[prediction13]: ./predictions/13.JPG
[prediction17]: ./predictions/17.JPG
[prediction25]: ./predictions/25.JPG
[prediction27]: ./predictions/27.JPG
[prediction38]: ./predictions/38.JPG
[prediction40]: ./predictions/40.JPG
[layer]: ./layer_visualization.JPG

[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

* Number of training examples = 34799
* Number of testing examples = 12630
* Image data shape = (32, 32, 3)
* Number of classes = 43

Here is a bar chart that summarizing the training data.
![alt text][image1]

### Data Preprocessing

The images were normalized using numpy. After looking at the memory usage, I realized that the normalized data was taking up around a gigabyte of memory. Therefore, I moved the normalization of the data to tensorflow. This reduced the amount of system memory allocated by about a gigabyte and seemed to speed up training, but I did not get any numbers on this.

I wanted to created a training pipeline that was closer to what would be used in production. Therefore, I used tf.contrib.data.Dataset to create datasets for training and inference. I was planning on using these for configuring a data augmentation pipeline, but I was able to get higher that 93% validation set accuracy without augmentation. 

### Network

For the network, I started out with the LeNet network. Since I had just figured out how to convert fully connected layers to convolutional layers, I converted the fully connected layers to convolutions. Although, I did not run it on larger than 32x32 images.
I trained the network, but the training accurracy was low. Since there were now 3 channels, I just decided to triple the number of filters in the convolutional layers and doubled them in the fully connected layers. Now, when I trained the network, it would overfit. Therefore, I added dropout to every layer except the first and last. Leading to the network below.


|Layer                  |Description                                      | 
|:---------------------:|:-----------------------------------------------:| 
|Input         		       | 32x32x3 tf.uint8							                         | 
|Normalize     	        | outputs 32x32x3 tf.float32  	                   |
|Convolution 5x5     	  | 1x1 stride, valid padding, outputs 28x28x(3x6) 	|
|RELU					              |												                                     |
|Max pooling 2x2	      	| 2x2 stride, outputs 14x14x18   				             |
|Convolution 5x5     	  | 1x1 stride, valid padding, outputs 10x10x(3x16) |
|RELU					              |												                                     |
|Max pooling 2x2	      	| 2x2 stride, outputs 5x5x48   				               |
|Drop out	      	       | 0.5 keep probility  				                        |
|Convolution 5x5     	  | 1x1 stride, valid padding, outputs 1x1x(2x240) 	|
|RELU					              |												                                     |
|Drop out	      	       | 0.5 keep probility  				                        |
|Convolution 1x1     	  | 1x1 stride, valid padding, outputs 1x1x(2x84) 	 |
|RELU					              |												                                     |
|Drop out	      	       | 0.5 keep probility  				                        |
|Convolution 1x1     	  | 1x1 stride, valid padding, outputs 1x1x43 	     |
|Reduce mean				        | outputs 43 
|Softmax				            | 

My final model results were:
* training set accuracy of 0.997
* validation set accuracy of 0.966
* test set accuracy of 0.954

### Inference

I tested the network on 12 images of German traffic signs that I found on the web. The accurracy was 0.833 on these images. Below are five of the images with their predictions.

I thought the network might have a problem with this sign. It is rotated, none of the training data is rotated, and there are not that many training examples for this sign. Although, the prediction was good.
![alt text][prediction0] 

The network was way off on this sign. It could be because it does not have a white border or the thick post. 
![alt text][prediction1]

You could imagine the network having a problem on this sign, because the training set contains several similiar signs.
![alt text][prediction25]

The network did not predict this sign correctly, but it was the second runner up and similiar to prediction.
![alt text][prediction27] 

The network had no doubt about this one and several other that were unique in the training set.
![alt text][prediction40]



###  Visualizing the Neural Network
Here is the visualization of the first layer activations for one of the test internet images. It interesting to note that the back ground has been completely eliminated and the network is just focusing on the sign. Also, it looks like there might be filtering by color and gradients going on.

![alt text][layer]


