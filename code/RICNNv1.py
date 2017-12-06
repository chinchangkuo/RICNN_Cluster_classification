import tensorflow as tf
import	 matplotlib.pyplot	as	plt
import numpy as np
import os
import cv2
import random
##------------------------------------------------------------------------

#train_folder_dir = 'CNN_demo\\CNN_l3_s3\\CNNinput\\'
#test_folder_dir = 'CNN_demo\\CNN_l3_s3\\CNNtest\\'
## for running in windows

train_folder_dir = 'CNN_demo/CNN_l3_s3/CNNinput/'
test_folder_dir = 'CNN_demo/CNN_l3_s3/CNNtest/'

n_classes = 24
hm_epochs = 15
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

c1w_size = 40
c2w_size = 20
c3w_size = 10
## convolution window size

all_img = [i for i in os.listdir(train_folder_dir) if i.endswith(".png")]
x = tf.placeholder('float')
y = tf.placeholder('float')
rot_steps = 72
ang_steps = 360/rot_steps
pick_size = 100

def image_preprocess_train(img_name, rot_steps, ang_steps):
    label = int(img_name[img_name.find('[')+1:img_name.find(']')])
    img_path = train_folder_dir + img_name
    img = cv2.imread(img_path,0)
    img = (np.amax(img)-img)/(np.amax(img)-np.amin(img))
    mirr_img = cv2.flip(img, 0)
    rows,cols = img.shape
    x_train = np.empty([rot_steps*2,160,160])
    y_train = np.zeros([rot_steps*2,n_classes])
    for k in range(2):
        for j in range(rot_steps):
            M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_steps*j,1)
            if k == 0: 
                dst = cv2.warpAffine(img,M,(cols,rows))
            else:
                dst = cv2.warpAffine(mirr_img,M,(cols,rows))
            r_img = dst[45:-45,45:-45]
            x_train[j + k*rot_steps] = r_img
            y_train[:,label-1] = 1
                
    return x_train , y_train

# for generating the training set


def image_random_pick(all_img, rot_steps, ang_steps, pick_size):
    x_pick = np.empty([pick_size,160,160])
    y_pick = np.zeros([pick_size,n_classes])
    for i in range(pick_size):
        seed = random.randrange(23)
        img_name = all_img[seed]
        label = int(img_name[img_name.find('[')+1:img_name.find(']')])
        img_path = train_folder_dir + img_name
        img = cv2.imread(img_path,0)
        img = (np.amax(img)-img)/(np.amax(img)-np.amin(img))
    
        mirr = random.randrange(2)
        if mirr == 1:
            img = cv2.flip(img, 0)
    
        rot = random.randrange(rot_steps)
        rows,cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_steps*rot,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        x_pick[i] = dst[45:-45,45:-45]
        y_pick[i, label-1] = 1

    return x_pick , y_pick

#for generating the testing set

def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides = [1,1,1,1], padding = 'SAME')

def maxpool2d(x, psize):
    return tf.nn.max_pool(x, ksize = [1,psize,psize,1], strides = [1,psize,psize,1], padding = 'SAME')
    # psize x psize pooling window for pooling.

def act_fun(x):
    return tf.nn.relu(x)

def rotate(x ,W):
    return tf.contrib.image.rotate(x , W) 
    
def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([c1w_size,c1w_size,1,16])),
               'W_conv2':tf.Variable(tf.random_normal([c2w_size,c2w_size,16,32])),
               'W_conv3':tf.Variable(tf.random_normal([c3w_size,c3w_size,32,64])),
               'W_fc1':tf.Variable(tf.random_normal([40*40*64,2048])),
               'W_fcr':tf.Variable(tf.random_normal([2048,2048])),
               'out':tf.Variable(tf.random_normal([2048,n_classes]))}
    
    biases =  {'b_conv1':tf.Variable(tf.random_normal([16])),
               'b_conv2':tf.Variable(tf.random_normal([32])),
               'b_conv3':tf.Variable(tf.random_normal([64])),
               'b_fc1':tf.Variable(tf.random_normal([2048])),
               'b_fcr':tf.Variable(tf.random_normal([2048])),
               'out':tf.Variable(tf.random_normal([n_classes]))}
    
    x = tf.reshape(x, shape = [-1, 160, 160, 1])
    conv1 = act_fun(conv2d(x,weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1, 2)
    
    conv2 = act_fun(conv2d(conv1,weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2, 2)
    
    conv3 = act_fun(conv2d(conv2,weights['W_conv3']) + biases['b_conv3'])  
    
    flat_conv3 = tf.reshape(conv3,[-1,40*40*64])
    fc1 = act_fun(tf.matmul(flat_conv3,weights['W_fc1'])+ biases['b_fc1'])
    fc2 = act_fun(tf.matmul(fc1,weights['W_fcr'])+ biases['b_fcr'])
    
    output =  tf.matmul(fc2, weights['out']) + biases['out']
    
    regularizers = tf.nn.l2_loss(weights['W_conv1']) + \
                   tf.nn.l2_loss(weights['W_conv2']) + \
                   tf.nn.l2_loss(weights['W_conv3']) + \
                   tf.nn.l2_loss(weights['W_fc1']) + \
                   tf.nn.l2_loss(weights['W_fcr']) + \
                   tf.nn.l2_loss(weights['out']) 
    
    return output, regularizers
    
def train_neural_network():
    prediction, regularizers = convolutional_neural_network(x)
    softmax = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    beta = 0.01
    cost = tf.reduce_mean(softmax + beta * (regularizers))
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, os.path.join(os.getcwd(), "save_model_RICNN.ckpt"))
        epoch_plt = []
        acc_train_plt = []
        acc_test_plt = []
        loss_plot = []
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in all_img:
                x_train, y_train = image_preprocess_train(i,rot_steps, ang_steps)
                epoch_x = x_train
                epoch_y = y_train
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            epoch_plt.append(epoch)
            loss_plot.append(epoch_loss)                        
            
            x_pick , y_pick = image_random_pick(all_img, rot_steps, ang_steps, pick_size)
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            acc, corr, p_max ,y_max = sess.run([accuracy,correct, tf.argmax(prediction,1),tf.argmax(y,1)], feed_dict={x: x_pick, y: y_pick})
            print ('accuracy for train set',acc)
            acc_train_plt.append(acc)
            
            x_train_test, y_train_test = image_random_pick(all_img, 720, 0.5, pick_size)
            acc_train_test, corr_train_test, p_max_train_test ,y_max_train_test  = sess.run([accuracy,correct, tf.argmax(prediction,1),tf.argmax(y,1)], feed_dict={x: x_train_test, y: y_train_test})
            print ('accuracy for test set', acc_train_test)
            acc_test_plt.append(acc_train_test)

        save_path = saver.save(sess, os.path.join(os.getcwd(), "save_model_RICNN.ckpt"))        
            
        plt.scatter(epoch_plt, acc_train_plt,color='r',s = 150, label="accuracy_train")
        plt.scatter(epoch_plt, acc_test_plt, color='b',s = 150, label="accuracy_test")
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)
        plt.xlabel('epoch')
        plt.xlim(-0.5, 14.5)
        plt.legend(loc='upper left',)
        plt.show()

train_neural_network()    
