import tensorflow as tf
import	 matplotlib.pyplot	as	plt
#import matplotlib.image as mpimg
#from sklearn import datasets
import numpy as np
import os
import cv2
import random
from collections import Counter
##------------------------------------------------------------------------

#train_folder_dir = 'CNN_demo\\CNN_l3_s3\\CNNinput\\'
#test_folder_dir = 'CNN_demo\\CNN_l3_s3\\CNNtest\\'
## for running in windows

train_folder_dir = 'CNN_demo/CNN_l3_s3/CNNinput/'
test_folder_dir = 'CNN_demo/CNN_l3_s3/CNNtest/'

n_classes = 24
hm_epochs = 25
keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

c1w_size = 8
c2w_size = 5
c3w_size = 3

## convolution window size
all_img = [i for i in os.listdir(train_folder_dir) if i.endswith(".png")]
x = tf.placeholder('float')
y = tf.placeholder('float')
g_step = tf.Variable(0, trainable=False)
learning_rate = 0.001
#learning_rate = tf.train.exponential_decay(learning_rate, g_step,
#                                            460, 0.9, staircase=True)


rot_steps = 72
ang_steps = 360/rot_steps
pick_size = 100


def move_to_com(img):
    m_img = cv2.moments(img)
    m_y=m_img['m01']/m_img['m00']
    m_x=m_img['m10']/m_img['m00']
    m_center = img.shape[0]/2
    m_move = np.float32([[1,0,m_center-m_x],[0,1,m_center-m_y]])
    dst = cv2.warpAffine(img,m_move,img.shape)
    mm_img = cv2.moments(dst)
    mm_y=mm_img['m01']/mm_img['m00']
    mm_x=mm_img['m10']/mm_img['m00']
    
    return img
 
#    img = cv2.imread(img_path,0)
#    cimg1 = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#    cv2.circle(cimg1,(int(m_center),int(m_center)),25,(255,0,0),2); 
#    cv2.circle(cimg1,(int(m_x),int(m_y)),25,(0,255,0),2);
#    cv2.circle(cimg1,(int(mm_x),int(mm_y)),25,(0,0,255),2); 
#    plt.imshow(cimg1)

def image_preprocess_train(img_name, rot_steps, ang_steps):
    label = int(img_name[img_name.find('[')+1:img_name.find(']')])
    img_path = train_folder_dir + img_name
    img = cv2.imread(img_path,0)
    img = (np.amax(img)-img)/(np.amax(img)-np.amin(img))
    img = move_to_com(img)
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
            
#    f, axarr = plt.subplots(2, rot_steps, figsize=(10, 3))
#    axarr[0,0].set_ylabel('Normal')
#    for i in range(rot_steps):
#        axarr[0, i].imshow(x_train[i], cmap="gray")
#        axarr[0, i].set_xticks([])
#        axarr[0, i].set_yticks([])
#        axarr[0, i].set_title(ang_steps*i )
#    axarr[1,0].set_ylabel('Mirror')
#    for i in range(rot_steps):
#        axarr[1, i].imshow(x_train[i+rot_steps],cmap="gray")
#        axarr[1, i].set_xticks([])
#        axarr[1, i].set_yticks([])
#    plt.show()
    
    return x_train , y_train

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
        img = move_to_com(img)
        mirr = random.randrange(2)
        if mirr == 1:
            img = cv2.flip(img, 0)
    
        rot = random.randrange(rot_steps)
        rows,cols = img.shape
        M = cv2.getRotationMatrix2D((cols/2,rows/2),ang_steps*rot,1)
        dst = cv2.warpAffine(img,M,(cols,rows))
        x_pick[i] = dst[45:-45,45:-45]
        y_pick[i, label-1] = 1
#        
#    f, axarr = plt.subplots(2, int(pick_size/2), figsize=(10, 4))
#    for i in range(int(pick_size/2)):
#        axarr[0, i].imshow(x_pick[i], cmap="gray")
#        index, = np.where(y_pick[i] == 1)
#        ylabel = int(index) + 1
#        axarr[0, i].set_title(ylabel)
#        axarr[0, i].set_xticks([])
#        axarr[0, i].set_yticks([])
#    for i in range(int(pick_size/2)):
#        axarr[1, i].imshow(x_pick[i+int(pick_size/2)],cmap="gray")
#        index, = np.where(y_pick[i+int(pick_size/2)] == 1)
#        ylabel = int(index) + 1
#        axarr[1, i].set_title(ylabel)
#        axarr[1, i].set_xticks([])
#        axarr[1, i].set_yticks([])
#    plt.show()        

    return x_pick , y_pick

def image_test(test_folder_dir):
    all_test_img = [i for i in os.listdir(test_folder_dir) if i.endswith(".png")]
    x_test = np.empty([len(all_test_img),160,160])
    y_test = np.zeros([len(all_test_img),n_classes])
    for n , i in enumerate(all_test_img):
        label = int(i[i.find('[')+1:i.find(']')])
        img_path = test_folder_dir + i
        img = cv2.imread(img_path,0)
        img = (np.amax(img)-img)/(np.amax(img)-np.amin(img))
        img = move_to_com(img)
        x_test[n] = img[45:-45,45:-45]
        y_test[n, label-1] = 1
    
    return x_test, y_test 



def conv2d(x,W):
    return tf.nn.conv2d(x,W, strides = [1,1,1,1], padding = 'SAME')

def maxpool2d(x, psize):
    return tf.nn.max_pool(x, ksize = [1,psize,psize,1], strides = [1,psize,psize,1], padding = 'SAME')
    #The ksize parameter is the size of the pooling window.
    #In our case, we're choosing a 2x2 pooling window for pooling.

def act_fun(x):
    #return tf.sigmoid(x)
    return tf.nn.relu(x)

def rotate(x ,W):
    return tf.contrib.image.rotate(x , W) 
    
def convolutional_neural_network(x):
    weights = {'W_conv1':tf.Variable(tf.random_normal([c1w_size,c1w_size,1,16])),
               'W_conv2':tf.Variable(tf.random_normal([c2w_size,c2w_size,16,32])),
               'W_conv3':tf.Variable(tf.random_normal([c3w_size,c3w_size,32,64])),
               'W_fc1':tf.Variable(tf.random_normal([20*20*64,2048])),
               'W_fc2':tf.Variable(tf.random_normal([2048,4096])),
               'out':tf.Variable(tf.random_normal([4096,n_classes]))}
    
    biases =  {'b_conv1':tf.Variable(tf.random_normal([16])),
               'b_conv2':tf.Variable(tf.random_normal([32])),
               'b_conv3':tf.Variable(tf.random_normal([64])),
               'b_fc1':tf.Variable(tf.random_normal([2048])),
               'b_fc2':tf.Variable(tf.random_normal([4096])),
               'out':tf.Variable(tf.random_normal([n_classes]))}
    
    x = tf.reshape(x, shape = [-1, 160, 160, 1])
    x = maxpool2d(x, 2)
    conv1 = act_fun(conv2d(x,weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1, 2)
    
    conv2 = act_fun(conv2d(conv1,weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2, 2)
    
    conv3 = act_fun(conv2d(conv2,weights['W_conv3']) + biases['b_conv3'])  
    
    flat_conv3 = tf.reshape(conv3,[-1,20*20*64])
    fc1 = act_fun(tf.matmul(flat_conv3,weights['W_fc1'])+ biases['b_fc1'])
    fc2 = act_fun(tf.matmul(fc1,weights['W_fc2'])+ biases['b_fc2'])

    #fcr = tf.nn.dropout(fcr,keep_rate)
    
    output =  tf.matmul(fc2, weights['out']) + biases['out']
    
    regularizers = tf.nn.l2_loss(weights['W_conv1']) + \
                   tf.nn.l2_loss(weights['W_conv2']) + \
                   tf.nn.l2_loss(weights['W_conv3']) + \
                   tf.nn.l2_loss(weights['W_fc1']) + \
                   tf.nn.l2_loss(weights['out']) 
    
    #rot_loss = tf.nn.l2_loss(tf.abs(fcr[0] - tf.reduce_mean(fcr, 0)))
    #rot_loss = tf.nn.l2_loss(tf.abs(fcr[0] - tf.reduce_mean(fcr, 0)))
    #rot_loss = tf.reduce_mean(tf.square(fcr[0] - fcr), 0)
    
    rot_loss = tf.reduce_mean(tf.square(fc2[0] - fc2))
    
    #rot_loss = tf.reduce_mean(tf.square(output[0] - output))
    
    return output, regularizers, rot_loss, fc2
    
def train_neural_network():
    #prediction = convolutional_neural_network(x)    
    #cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    prediction, regularizers, rot_loss, fcr = convolutional_neural_network(x)
    softmax = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y) )
    beta = 0.01
    #gama = 0.0005
    gama = 0
    #cost = tf.reduce_mean(softmax + beta * (regularizers))
    #cost = tf.reduce_mean(softmax + beta * (regularizers) + (gama/(rot_steps*2)) * rot_loss )
    cost = tf.reduce_mean(softmax + beta * (regularizers) + (gama/2) * rot_loss )
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=g_step)
    
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, os.path.join(os.getcwd(), "save_model_RICNN.ckpt"))
        epoch_plt = []
        acc_train_plt = []
        acc_validate_plt = []
        acc_test_plt = []
        loss_plot = []
        for epoch in range(hm_epochs):
            epoch_loss = 0
            for i in all_img:
                x_train, y_train = image_preprocess_train(i,rot_steps, ang_steps)
                epoch_x = x_train
                epoch_y = y_train
                #_, c, fcr_out,fcr_sq,rot_loss_out = sess.run([optimizer, cost, fcr, tf.square(fcr[0] - fcr),rot_loss ], feed_dict={x: epoch_x, y: epoch_y})
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of',hm_epochs,'loss:',epoch_loss)
            epoch_plt.append(epoch)
            loss_plot.append(epoch_loss)                        
            #print ((fcr_out[0] - fcr_out)[2][3:5], fcr_sq[2][3:5],rot_loss_out )
            x_pick , y_pick = image_random_pick(all_img, rot_steps, ang_steps, pick_size)
            correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            acc, corr, p_max ,y_max,gg_step = sess.run([accuracy,correct, tf.argmax(prediction,1),tf.argmax(y,1),g_step], feed_dict={x: x_pick, y: y_pick})
            print ('accuracy for train set', acc, gg_step)
            acc_train_plt.append(acc)
            
            x_train_test, y_train_test = image_random_pick(all_img, 720, 0.5, pick_size)
            acc_train_test, corr_train_test, p_max_train_test ,y_max_train_test  = sess.run([accuracy,correct, tf.argmax(prediction,1),tf.argmax(y,1)], feed_dict={x: x_train_test, y: y_train_test})
            print ('accuracy for validate set', acc_train_test)
            acc_validate_plt.append(acc_train_test)
            
            x_test, y_test = image_test(test_folder_dir)
            acc_test, corr_test, p_max_test ,y_max_test  = sess.run([accuracy,correct, tf.argmax(prediction,1),tf.argmax(y,1)], feed_dict={x: x_test, y: y_test})
            print ('accuracy for test set', acc_test)
            acc_test_plt.append(acc_test)
        #print (p_max_test ,y_max_test)
        wrong_list=[]
        for i,j in enumerate(p_max_test):
            if j != y_max_test[i]:
                str(j)+':'+str(y_max_test[i])
                wrong_list.append(str(y_max_test[i])+'to'+str(j))
        print(Counter(wrong_list).most_common())
            
        #print (corr_test)
        #print (p_max_test)
        #print (y_max_test)
        #save_path = saver.save(sess, os.path.join(os.getcwd(), "save_model_RICNN.ckpt"))        
#        
#        print(correct.eval(feed_dict={x:x_train, y:y_train}))
#        accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#        print(prediction.eval(feed_dict={x:x_train}))
#        print(tf.argmax(prediction,1).eval(feed_dict={x:x_train}))
#        print(tf.argmax(y,1).eval(feed_dict={y:y_train}))
#        print(y_train[0])
#        print(x_train.shape)
#        print(y_train.shape)
#        print('Accuracy:',accuracy.eval(feed_dict={x:x_train, y:y_train}))
##---------------------------------------------------------------------------
            
        plt.scatter(epoch_plt, acc_train_plt,color='r',s = 150, label="accuracy_train")
        plt.scatter(epoch_plt, acc_validate_plt, color='b',s = 150, label="accuracy_validate")
        plt.scatter(epoch_plt, acc_test_plt, color='g',s = 150, label="accuracy_test")
        plt.ylabel('Accuracy')
        plt.ylim(0, 1.05)
        plt.xlabel('epoch')
        plt.xlim(-0.5, (hm_epochs-0.5))
        plt.legend(loc='upper left')
        #plt.legend()
        plt.show()
        #plt.plot(epoch_plt,acc_train_plt,'bs',epoch_plt, acc_test_plt, 'rc')
        #plt.set_ylabel('Accuracy')
        #plt.set_ylabel('epoch')
        #plt.show()
#--------------------for generating the seed images
#    all_img = [i for i in os.listdir(train_folder_dir) if i.endswith(".png")]
#    f, axarr = plt.subplots(5, 5, figsize=(4, 4))    
#    for i in range(5):
#        for j in range(5):
#            if i + j*5 < 23:
#                img_path = train_folder_dir + all_img[i + j * 5]
#                img = cv2.imread(img_path,0)
#                img = (np.amax(img)-img)/(np.amax(img)-np.amin(img))            
#                axarr[ j, i].imshow(img, cmap="gray")
#            axarr[j, i].set_xticks([])
#            axarr[j, i].set_yticks([])
#            axarr[j,i].axis('off')

            
            
        
#train_neural_network(x, y, label)
train_neural_network()    
