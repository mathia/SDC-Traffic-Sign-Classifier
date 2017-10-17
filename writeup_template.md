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

[image1]: ./writeup_images/sign_class_examples.png "Example Image for Each Class"
[image2]: ./writeup_images/training-count-by-class-histogram.png "Histogram of Counts by Class ID"
[image3]: ./writeup_images/validation-count-by-class-histogram.png "Histogram of Counts by Class ID"
[image4]: ./writeup_images/test-count-by-class-histogram.png "Histogram of Counts by Class ID"
[image5]: ./writeup_images/synthesized-example.png "Example of Synthesized Images"
[image6]: ./writeup_images/new_images.png "New Traffic Sign Images"

[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/mathia/SDC-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the Numpy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799 images
* The size of the validation set is 4,410 images
* The size of test set is 12,630 images
* The shape of a traffic sign image is 32 x 32 x 3
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set.
Below is an example image for each of the 43 classes of signs.

![Example Images][image1]

Histograms of the number of images for each sign class were generated for the training, validation and test data sets.
![Training Histogram][image2]
![Validation Histogram][image3]
![Test Histogram][image4]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

The above histograms show the traing data set has some classes with significantly more examples than others.  To prevent the training data from biasing the model towards the classes that are more prevelent, I augmented the training data set by generating new training examples for underrepresented classes.  The new images were synthesized by rotating, translating and perspective warping images from the original training data set by small random amounts.  Any class that had less than 750 images in the original training set was augmented to bring it to 750 images.

![Synthesized Example][image5]

All input image data has the color channel values normalized to improve the optimizers performance during training.  The goal was to make the pixel values for each color channel have a zero mean and unit variance but for simplicity's sake the actually mean was not calculated from the training set values, it was assumed to be 128 as the color range is 0-255.  All data sets were randomly shuffled as the supplied data sets were ordered by sign class.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer        		|     Description     				| 
|:---------------------:|:---------------------------------------------:| 
| Input        		| 32x32x3 RGB image   				| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6			|
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16	|
| RELU			|						|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16			|
| Flatten		| Outputs 400 x 1				|
| Fully connected	| Outputs 120  					|
| RELU			| Dropout with .5 keep_prob during training	|
| Fully connected	| Outputs 43  					|
| RELU			| Dropout with .5 keep_prob during training	|
| Softmax		| 	   					|
 


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the Adam optimizer.  I did not adjust the learning rate as the Adam varies the learning rate as part of the algorithm and I found my initial values of 0.001 seemed to work well.  The batch size is 128 and 25 epochs were used during the final training run.  I found that more epochs did not help performance on the validation set.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 99.2%
* validation set accuracy of 95.7%
* test set accuracy of 94.2%

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
  Initially used the LeNet architecture as was implemented in TensorFlow from the LeNet class lab where it was applied to the MNIST data set.  The LeNet architecture was first described in the following paper where it was applied to same GTSRB data set this project is working with so it is a proven model:  [http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf]
 
* What were some problems with the initial architecture?
  The model was overfitting as training accuracy approached 100% but validation was around 91%.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
  Dropout was applied to the two fully connected layers to combat overfitting.  Additionally the data set was augmented to make the number of examples per class more even to help the model generalize.

* Which parameters were tuned? How were they adjusted and why?
  The number of epochs was adjusted.  Different values from 10 to 100 were tried.  Plots of the training and validation loss vs number of epochs was used to determine when the model was no longer learning parameters that generalized well.  

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
  The standard LeNet model was able to fit the training data with almost 100% accuracy and the training data set was fairly large so the model seemed sufficiently complex to capture the features in the data set, but it wasn't generalizing well.  Dropout is a good way encourage the model to learn more generalized features.

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![New Images][image6]

These might be difficult to classify because they are more closely zoomed in than the training data images.  Also, while these images are more easily distinguished to the human eye than the training images (brighter, less noise), the fact that they are different from the training data might give the model trouble.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image		        |     Prediction	  			| 
|:---------------------:|:---------------------------------------------:| 
| No Vehicles  		| No Vehicles   				| 
| Speed limit 30  	| Speed limt 20 				|
| Ahead only		| Ahead only					|
| Go straight or left   | Go straight or left				|
| Bumpy Road		| Bumpy Road 					|
| General caution	| General caution				|

The model was able to correctly guess 5 of the 6 traffic signs, which gives an accuracy of 83%. This compares favorably to the accuracy on the test set of 94%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

All of the prodictions show very high probability (> 98%) for the predicted sign ID, even the one that is incorrect.  The incorrect classification does have the correct ID as the 2nd most likely, but it is only 1%.

Image #0
  logit ranked #1:
     Probablity: 0.9962067604064941
     Sign ID: 15
  logit ranked #2:
     Probablity: 0.0016332280356436968
     Sign ID: 32
  logit ranked #3:
     Probablity: 0.0011647774372249842
     Sign ID: 13
  logit ranked #4:
     Probablity: 0.000438211573055014
     Sign ID: 39
  logit ranked #5:
     Probablity: 0.00023985384905245155
     Sign ID: 26
Image #1
  logit ranked #1:
     Probablity: 0.9884748458862305
     Sign ID: 0
  logit ranked #2:
     Probablity: 0.011522348038852215
     Sign ID: 1
  logit ranked #3:
     Probablity: 2.528830464143539e-06
     Sign ID: 28
  logit ranked #4:
     Probablity: 3.891424285029643e-07
     Sign ID: 27
  logit ranked #5:
     Probablity: 2.7704800231731497e-08
     Sign ID: 24
Image #2
  logit ranked #1:
     Probablity: 0.9997001886367798
     Sign ID: 35
  logit ranked #2:
     Probablity: 0.00013589684385806322
     Sign ID: 34
  logit ranked #3:
     Probablity: 6.764916179236025e-05
     Sign ID: 33
  logit ranked #4:
     Probablity: 5.426889401860535e-05
     Sign ID: 36
  logit ranked #5:
     Probablity: 3.834455128526315e-05
     Sign ID: 37
Image #3
  logit ranked #1:
     Probablity: 1.0
     Sign ID: 37
  logit ranked #2:
     Probablity: 3.967610094635088e-12
     Sign ID: 40
  logit ranked #3:
     Probablity: 5.068698824377271e-13
     Sign ID: 39
  logit ranked #4:
     Probablity: 2.494627226144741e-14
     Sign ID: 33
  logit ranked #5:
     Probablity: 7.266077285287971e-20
     Sign ID: 35
Image #4
  logit ranked #1:
     Probablity: 0.9999974966049194
     Sign ID: 22
  logit ranked #2:
     Probablity: 2.502274583093822e-06
     Sign ID: 29
  logit ranked #3:
     Probablity: 3.472740983667677e-09
     Sign ID: 26
  logit ranked #4:
     Probablity: 6.826916809643535e-11
     Sign ID: 25
  logit ranked #5:
     Probablity: 3.3514751846160096e-13
     Sign ID: 15
Image #5
  logit ranked #1:
     Probablity: 0.9999996423721313
     Sign ID: 18
  logit ranked #2:
     Probablity: 4.168192617726163e-07
     Sign ID: 27
  logit ranked #3:
     Probablity: 1.4886928356716567e-09
     Sign ID: 26
  logit ranked #4:
     Probablity: 3.319242864114805e-16
     Sign ID: 24
  logit ranked #5:
     Probablity: 9.225707705999131e-19
     Sign ID: 11

### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


