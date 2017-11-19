## **Advanced Lane Finding**

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[fit]: ./writeup_images/fit.JPG "Fit"
[mask]: ./writeup_images/mask.JPG "Mask"
[process]: ./writeup_images/process.JPG "Process"
[sliding]: ./writeup_images/sliding.JPG "Sliding"
[transform]: ./writeup_images/transform.JPG "Transform"
[undistort]: ./writeup_images/undistort.JPG "Undistort"
[points]: ./writeup_images/points.JPG "Points"


[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## Rubric Points

Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/571/view) individually and describe how I addressed each point in my implementation.  
---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is located under Camera Calibration in the notebook AdvancedLaneLines.ipynb. The function calibrate camera calculates the camera matrix and distortion coefficients used by the function undistort. It first finds the checker board corners for all 20 of the calibration images using cv2.findChessboardCorners. Then, it uses the corners with cv2.calibrateCamera to create the matrix/coeffiecients. 

The function undistort uses the matrix/coeffiecients to undistort an image using cv2.undistort. This is the function that is used in lane finding pipeline. Below is an example of undistort applied to one of the checkerboard calibration images.

![alt text][undistort]

### Pipeline (single images)

### 1. Provide an example of a distortion-corrected image.

Appying the undistort function is the first step in the pipeline. Above is an example of it applied to one of the checkerboard calibration images.

### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

The code for this step is located under Gradient and Color Masks in the notebook AdvancedLaneLines.ipynb. I created various functions to create masks (i.e binary thresholded images). This included gradient, gradient direction, and hls colorspace masks. Also, I created function to create the conjunction and disjunction of mask. I tried various combinations of mask, but finally settled on a hls colorspace saturation mask. It seemed to work ok, and let me focus on other parts of the project. 

The mask function is the second step in the pipeline and applies the saturation mask. Below is and example of it applied to a test image.

![alt text][mask]

### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for this step is located under Perspective Transform in the notebook AdvancedLaneLines.ipynb. The crux of creating the perspective transform was determing the source rectangle. I found the source rectangle using an undistorted version of straight_lines1.jpg. Then, I created functions transform and reverse an image.

![alt text][points]

Since the transform function determines how far out we are trying to find lane lines, I added the alpha factor to scale the length of the rectangle. Trying to get the pipeline to work, I set alpha to 0.1 to scale down how far out I was looking to get ride of some horizon artifacts. This probably would not have been needed if I was using a better mask. Below is an example of applying the transform function.

![alt text][transform]

### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
The code for this step located under Fitting Lines in the notebook AdvancedLaneLines.ipynb. I used the code from class to create a sliding_windows_fit function. Here is an example of it's use.

![alt text][sliding]

I then created the fit function using code from class to fit new lines base on previous fit lines. Here is an example of it's use.

![alt text][fit]

### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I create a function calculate_curvature under Fitting Lines in the notebook AdvancedLaneLines.ipynb using the method outlined in class. 
I then modifed my fit function to also calculated the curvature.

### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implmented the process_image to perform the whole pipeline on a frame and keep track of previous frames. It is located under Frame Processing in the notebook AdvancedLaneLines.ipynb. Below is an example of it being applied to an image.

![alt text][process]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
