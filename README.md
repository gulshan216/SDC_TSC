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

[image1]: ./examples/visual.png "Visualization"
[image2]: ./examples/preprocessed.png "Grayscaling"
[image3]: ./examples/augmented.png "Random Noise"
[image4]: ./examples/images_from_web.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"
[image9]: ./examples/distribution.png "Augmented Dataset Distribution"
[image10]: ./examples/recall_before_aug.png "Recall before Augmentation"
[image11]: ./examples/recall_after_aug.png "Recall after Augmentation"

### Data Set Summary & Exploration

#### 1. Basic Summary of the Dataset

I used the numpy library to calculate summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32,32,3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing number of examples for each of the traffic sign in the training data set.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Preprocessing of the image data

As a first step, I decided to convert the images to grayscale because most of the traffic sign images have similar color pattern and hence the color channels do not provide too much of an advantage.
Also the contrast of the sign images in the dataset is quite poor and hence global histogram equalization was applied on the grayscale image to provide better feature extraction. Also it provides better results than just normalizing the image pixel by pixel.
Here is an example of a traffic sign image before and after grayscaling+histogram equalization.

![alt text][image2]

Using just this as a preprocessing technique in combination Le-Net architecture  provides an accuracy of just 93.5%. Examining the distribution of examples per traffic sign in the training dataset we can see that many signs have less than 300 training examples.
Hence I decided to generate additional data using common data augmentation techniques such as rotation, translation and flip. With the augmented dataset 
Since flipping the traffic sign images might yield to an invalid sign or completely different sign in the dataset itself. Hence we classify traffic signs into 3 classes:
1) Traffic signs which remain same after vertical or horizontal axis flip
2) Traffic signs which remain same after vertical axis flip
3) Traffic signs which represent a different traffic sign after vertical axis flip
To augment the dataset, for each image, I add 10 images with a small but random rotation and translation to the original image and then preprocessed using the technique mentioned previously. Also if the image belongs to one of the flip traffic sign classes above the images are flipped accordingly, preprocessed and then added to the augmented dataset. 
Here is an example of an original image before and after rotation, translation:

![alt text][image3]

The distribution of examples per traffic sign in the augmented dataset can be seen below:
![alt text][image9]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 grayscale image						| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6  				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, outputs 5x5x16      				|
| Flatten       		|           									|
| Fully connected		| outputs 120   								|
| RELU  				|												|
| Fully connected		| outputs 84   									|
| RELU  				|												|
| Dropout       		| keep prob = 0.5								|
| Fully connected		| outputs 43									|
| Softmax  				|												|
 


#### 3. Training the model

To train the model, I used the Adam Optimizer as it provides much better convergence than the stochastic Gradient Descent optimizer. Also the below hyperparameters were used to provide better results:
Batch Size= 128
Epochs = 15
Learning rate = 0.001
Keep prob(for the Dropout layer) = 0.5

#### 4. Improving the accuracy of my classifier

My final model results were:
* training set accuracy of 98.5%
* validation set accuracy of 97.6%
* test set accuracy of 94.36%

The architecture chosen was the Le-Net architecture. This model was proved to work well in the recognition hand and print written character. It could be a good fit for the traffic sign classification.
In order to prevent the model from overfitting on the training set, I added a dropout layer with keep probability of 0.5 before the last fully connected layer. 
Before the addition of the dropout layer, the validation set accuracy achieved was just 90% and after adding the dropout layer validation set accuracy achieved was 93.5%.
Further using data augmentation techniques, helped achieve the validation set accuracy of 97.6%.
The final model accuracy on the training and validation set is good and close to each other and hence proving the model does not overfit too much on the training data. Also to get just a brief idea of the performance improvement of the model before and after data augmentation I plotted the recall rate for each of the traffic sign classes. Although we would still need to get the precision as well to get a complete picture of how the model is performing
Recall per class before Data augmentation
![alt text][image10]
Recall per class after Data augmentation
![alt text][image11]

### Testing the Model on New Images

#### 1. Choosing five random German traffic signs found on the web

Here are five German traffic signs that I found on the web:

![alt text][image4]

The first image might be difficult to classify because ...

#### 2. The model's predictions on these new traffic signs.

Here are the results of the prediction:

| Image			                        |     Prediction       					| 
|:-------------------------------------:|:-------------------------------------:| 
| Stop Sign      		                | Stop sign	        					| 
| No entry     			                | No entry  							|
| 100 km/h				                | 100 km/h								|
| Priority Road    		                | Priority Road			 				|
| Right-of-way at the next intersection | Right-of-way at the next intersection	|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. This compares favorably to the accuracy on the test set of 94.36% and also based on the fact that a lot of augmented data was used for training.

