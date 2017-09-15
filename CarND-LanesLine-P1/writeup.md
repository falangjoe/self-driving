# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

The pipeline is just the standard one from the exercise. It is convert to grayscale, smooth with gaussian kernel, canny edge detection, mask the region of interest, hough transform to create line segment, determine lane line from line segments, and draw the lines.
The non standard part was determining the lane lines. My first pipeline split the line segments into two groups by slope, determined the intercept at the bottom of the image, and drew two lines. This worked good for the sample videos and images, but failed on the challenge video.
Instead of drawing two lines, I really wanted to draw path that would take into account the curvature of the lanes. The pipleline I created first ordered the line segment by there height from the bottom of the image and got ride of any segments that were too perpendicular to the road.
Then, I iterated through line segments creating paths by connecting the next line segment to the path that made the most sense or created a new path. Then I groups the path by there intercept with the bottom of the image and pick two. This improved the performance on the challenge video, but the results were still not that great.


The pipeline has a lot of short comings. First, it needs to have a more robust way of pick which two paths of possible multiple one to choose. On the challenge video it would sometimes pick a path on the edge of the road and not the lane line. 
Also, the pipeline seemed to not detect any line segments when the car went into the shade and the color of the road changed.

There are many areas I would like to look at for possible improvement. First figure out how to get line segments to show up in shaded areas. Also, find a better measure of equivalency of paths that could measure the distance they are apart. This could allow picking out lane lines. Before any of these, I would probably want to do more research on what other have done.


