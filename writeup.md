# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

The video can be checked [here](https://github.com/feleir/behavioral-cloning/blob/master/video.mp4?raw=true)

[//]: # (Image References)

[center]: ./images/center.png "Center"
[right]: ./images/right.png "Right"
[left]: ./images/left.png "Left"
[flippedcenter]: ./images/flippedcenter.png "Flipped center"
[flippedright]: ./images/flippedright.png "Flipped right"
[flippedleft]: ./images/flippedleft.png "Flipped left"
[croppedcenter]: ./images/croppedcenter.png "Cropped center"
[croppedright]: ./images/croppedright.png "Cropped right"
[croppedleft]: ./images/croppedleft.png "Cropped left"
[rgbcenter]: ./images/rgbcenter.png "RGB center"
[rgbright]: ./images/rgbright.png "RGB right"
[rgbleft]: ./images/rgbleft.png "RGB left"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

On my first iterations I tried using [Lenet](http://yann.lecun.com/exdb/lenet/) first but after some testing ended up using the deep neural network described [here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model sumary is as follows:

| Layer                    | Output           |
|--------------------------|------------------|
| Lambda                   | (160, 320, 3)    |
| Cropping2D               | (90, 320, 3)     |
| Conv2D                   | (43, 158, 24)    |
| Conv2D                   | (20, 77, 36)     |
| Conv2D                   | (8, 37, 48)      |
| Conv2D                   | (6, 35, 64)      |
| Conv2D                   | (4, 33, 64)      |
| Flatten                  | 8448             |
| Dense                    | 100              |
| Dense                    | 50               |
| Dense                    | 10               |
| Dense                    | 1                |

The model includes RELU layers to introduce nonlinearity (code line 50 to 54), and the data is normalized in the model using a Keras lambda layer (code line 47). 

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 129). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track. But I did test create the model only with data coming from driving around the first track, due to time constraints I was unable to produce a model that will drive around the second track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 79).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. I used only data coming from the first track.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

On my first iterations I tried using [Lenet](http://yann.lecun.com/exdb/lenet/) first, using training data I created driving two laps around the first track with different ways of takes the curves, the car drive good enough on the straights but did not take the curves as expected and ended up in the like.

Tried a few things to improve but it was not enough
- Added a lambda step at the start of the pipeline to normalize the images
- Cropped the images to only take into account sections of the road that the car should care about
- Augmented the number of epochs and reduced the learning rate.

On the next steps I decided to use the deep neural network described by Nvidia [here](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

- Converted the images to RGB (model.py 31).
- Augmented the data by adding the same image flipped with a negative angle(model.py line 35). 
In addition to that, the left and right camera images where introduced with a correction factor on the angle to help the car go back to the lane(model.py - lines 93-95). 
- Added more training data, driving the track backwards.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road but only for the first track.

#### 2. Final Model Architecture

The final model architecture was described in the previous sections (model.py lines 44-61).

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior I recorded the following laps:
- One lap driving on the center lane
- Another one driving close to the right lane
- Another one driving on the opossite side, hence steering right instead
- Recorded some recovering situations so the car can handle some of them

The data used was coming from the center, right and left cameras:

| Left        | Center          | Right         |
|-------------|-----------------|---------------|
|![left][left]|![center][center]|![right][right]|

to gather more training data and prevent from steering angle bias, the images were flipped too:

| Left        | Center          | Right         |
|-------------|-----------------|---------------|
|![flippedleft][flippedleft]|![flippedcenter][flippedcenter]|![flippedright][flippedright]|

And converted to RGB which had an impact of the model training:

| Left        | Center          | Right         |
|-------------|-----------------|---------------|
|![rgbleft][rgbleft]|![rgbcenter][rgbcenter]|![rgbright][rgbright]|

to reduce training time the images were cropped to only feed important data to the model:

| Left        | Center          | Right         |
|-------------|-----------------|---------------|
|![croppedleft][croppedleft]|![croppedcenter][croppedcenter]|![croppedright][croppedright]|

I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 50 as evidenced by the changes in the validation loss during training. 
I used an adam optimizer so that manually training the learning rate wasn't necessary.