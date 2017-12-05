## **Vehicle Detection Project**

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[predictions]: ./writeup_images/predictions.JPG
[slidingwindows1]: ./writeup_images/slidingwindows1.JPG
[slidingwindows2]: ./writeup_images/slidingwindows2.JPG
[slidingwindows3]: ./writeup_images/slidingwindows3.JPG
[pipeline]: ./writeup_images/pipeline.JPG


## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/513/view) individually and describe how I addressed each point in my implementation.  
---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.
You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is located under Extracting Features in the notebook VehicleDetectionAndTracking.ipynb. Functions were created to extract HOG, spatial, and color histogram features. I also created a function that runs all three of these feature extractors and concatenates their output. This function was used both for training and sliding windows classification. When sliding windows classifiction is done, the precomputed HOG features are passed to function and not recomputed.

Moreover, I have a function to seperate training files into cars/notcar groups and a function to load the files/extract features for a groups of images.

![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

When I first started running the sliding windows detection, I wanted to have a tight fit for the bounding box around the car. Having a smaller pixel per cell and a lower cell per step allowed tighter bounding boxes, but there was a trade off. The smaller the shift in the sliding windows made for tighter bounding boxes, but also resulted in more false positives. Also, it made the process run slower. To get rid of the false positives, I had to make cell per step high and sacrifice the tight bounding boxes. I ended up with the following parameters.

pix_per_cell = 16
cell_per_block = 2
orientations = 9

This allowed me to reduce the HOG feature count, but still get the sliding window positions that I needed. I left the orientations to the value used in the lecture. 

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The code for this step is located under SVM training in the notebook VehicleDetectionAndTracking.ipynb. The traing function takes the get_image_features functions and returns a function to make predictions. Here are some examples of the prediction function being run.

![alt text][predictions]

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The code for this step is located under Sliding Windows in the notebook VehicleDetectionAndTracking.ipynb. I wanted the scales to be a constant multiple of the previous step and settled on 1, 2*1 = 2, and 2*2 = 4. Here are some tests of these scales.

![alt text][slidingwindows1]

![alt text][slidingwindows2]

![alt text][slidingwindows3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

![alt text][pipeline]

This shows my pipeline on one image with bounding boxes. I searched on three scales totaling 117 + 57 + 27 = 201 windows. Hog features alone work pretty good, but I had to add spatial color and color histograms to get rid of all false positives. Everything was done with the YUV color space.

---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./output_videos/project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I used the method presented in the lecture using heatmaps to create bounding boxes, with some changes. I created heatmaps for each scale and thresholded each one seperately. I did this so that the detections at different scales would not interfere so much with each other. Once each scale was thresholded, I saved the sum of these heatmaps per frame. The pipeline would find the mean of the heatmaps from the last 10 frames and use the threshold of the mean to create the bounding boxes. 

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

As you can see from the video, the pipeline is detecting cars going in the opposite direction on the other side of the highway. Since these are moving fast relative to the camera, the heatmaps are spread out and the bounding boxes do not track these cars well. Therefore, someway of tracking the velocity of the boxes relative to the camera is need. Also, the bounding boxes change shape and position in a non continuous way, but we know that the bounding boxes should move and change size in a continuous manner. Therefore by making changes to the size and position of a box in a more continuous way each frame could give better tracking of the cars. Moreover, the svm is probably overfitting the data, because of the time series issue. Training with better curated data could probably get better results and maybe allow the spatial and histogram features to be removed speeding up the pipeline.

