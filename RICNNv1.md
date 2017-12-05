# Preliminary Goal
•The Goal for this code is to set up a ground work for RICNN with the very limited number (23) of distinguished data as seed images.  

•Use seed images to train the model and evaluated the result to test the rotational invariance.  

# Training Data set
•The training Data in this code ifs from the same set of the data for the image processing demonstration:  https://github.com/chinchangkuo/ImageProcessing-bubble_cluster_n3l3s

•Adjust the intensity of images to be in between 0 and 1.
 RICNN_Cluster_classification/Figv1/seed_1.png 
 
![Training set](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/seed_1.png)

# Data Augment

•Rotate the image and the corresponding mirror image with a series of rotation steps. For example, the following figure is the result for 45 degree rotation step.
	
![Data Augment](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/augment_1.png)	

•For the actual training process, the rotation step is set to 5 degree, which generates 144 training images with 1 seed image.

# Testing Data generator

•In this code, the testing set has been generated randomly with the finer rotation step. For example, the following figure is the result for 16 randomly choosing testing image from the seed images with the random rotation. 

![Data Augment](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/test_1.png)	

•For the actual testing process, the rotation step is set to 0.5 degree, which generates 720 random rotations for both original and mirror images.

# RICNN 
•The structure for the current RICNN is as below:

Conv1 : 40 × 40  × 16

Conv2:  20 × 20 × 32

Conv3: 10 × 10 × 64

fc1: 2048

fc2: 2048

Output: 24 

•The trained convolution weights are visualized as follow:

Conv1

![Conv1](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/conv1.png)	

Conv2

![Conv2_1](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/conv2_1_1.png)	
![Conv2_2](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/conv2_1_2.png)	

Conv3

![Conv3_1](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/conv3_1_1.png)
![Conv3_2](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/conv3_1_2.png)
![Conv3_3](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/conv3_1_3.png)
![Conv3_4](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/conv3_1_4.png)
![Conv3_5](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/conv3_2_1.png)
![Conv3_6](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/conv3_2_2.png)
![Conv3_7](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/conv3_2_3.png)
![Conv3_8](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/conv3_2_4.png)

# Result

•The result for the accuracy of both the train set and testing set as a function of training epoch has be shown below:

![result](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv1/figure_1.png)

•The Accuracy saturated around 0.8 for the testing set, which suggests that the current model is somehow capable to handle 90% new  rotations that does not exist in the training set, but still have room to be improved.

