# Preliminary Goal
•The Goal for this code is to improve the accuracy for the valiadition and ensure the rotational invariance.

•In advance thet the testing accuracy by different input training set sizes.

# Modification
•The structure for the current RICNN is as below:

Input images: from 160 x 160 to 80 x 80

Conv1 : from 40 × 40  × 16 to 8 × 8  × 16 

Conv2:  from 20 × 20 × 32 to 5 × 5  × 32

Conv3: from 10 × 10 × 64 to 3 × 3  × 64

fc1: 2048

fc2: from 2048 to 4096

Output: 24   

•The image has been recenter to the center of mass

# Result
•The accuracy for the valiation set is close to 1 in general after about 15 epochs.

•The accuracy for the test set increases with the training set size:

![all set number](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv2/all.png)

![Accuracy set number](https://raw.githubusercontent.com/chinchangkuo/RICNN_Cluster_classification/master/Figv2/Figure_Acc_TrainingN.png)

•The accuracy for the test set with the training set size of 176 is close to 0.8, which means averagely every state needs at least 8 independent input images to reach the accuracy of 0.8.

•The next question wil beif the required input image set can be decrease in the case of focusing on some specific states that we are interested instead of all the possible states. 


