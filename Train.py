'''
Copyright (C) 2019 May 26 By JSH all rights reserved
Written by Sanghyeon Jo <josanghyeokn@gmail.com>
'''

import os, cv2
import numpy as np
import tensorflow as tf

from MLP import *
from Utils import *
from Define import *

mnist = MNIST_Download("../../../Dataset/MNIST/", one_hot = True)
iter_count = mnist.train.num_examples // BATCH_SIZE

input_var = tf.placeholder(dtype = tf.float32, shape = [None, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL])
label_var = tf.placeholder(dtype = tf.float32, shape = [None, CLASSES])

#0~9 -> (one hot)
# GT 0 : [1, 0, 0, 0, 0, 0, 0, 0, .., 0]
# GT 4 : [0, 0, 0, 0, 1, 0, 0, 0, ... 0]
# [ ? ? ? ? ? ?] -> [0.1, 0.6, 0.1, 0.2, 0.2, ..] = argmax() =  1.

# no activation
# net = MLP(input_var, activation = False)
# sigmoid activation
net = MLP(input_var, activation = True)

print(net)

loss_op = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits=net, labels=label_var) )

correct_prediction = tf.equal(tf.argmax(net, 1), tf.argmax(label_var, 1))
accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))

train_op = tf.train.AdamOptimizer(learning_rate=LEARNINT_RATE).minimize(loss_op)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    
    # train
    for epoch in range(MAX_EPOCH):

        losses = []
        for iter in range(iter_count):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)

            #[BATCH_SIZE, 784] -> (28, 28, 1)
            #[BATCH_SIZE, 10] 

            _, loss = sess.run([train_op, loss_op], 
                               feed_dict={input_var : batch_x, 
                                          label_var : batch_y})

            losses.append(loss)
            
        print("# epoch = {}, avg_loss = {}".format(epoch + 1, np.mean(losses)))
    
    # test
    accuracy = sess.run(accuracy_op, feed_dict = {input_var : mnist.test.images,
                                                  label_var : mnist.test.labels})
    
    print("# Test Accuracy = {}".format(accuracy))

'''
# activation sigmoid
# epoch = 1, avg_loss = 0.5996183156967163
# epoch = 2, avg_loss = 0.23397578299045563
# epoch = 3, avg_loss = 0.174993097782135
# epoch = 4, avg_loss = 0.13806749880313873
# epoch = 5, avg_loss = 0.11109480261802673
# epoch = 6, avg_loss = 0.09049661457538605
# epoch = 7, avg_loss = 0.07452381402254105
# epoch = 8, avg_loss = 0.061868805438280106
# epoch = 9, avg_loss = 0.05112959071993828
# epoch = 10, avg_loss = 0.04267793521285057
# epoch = 11, avg_loss = 0.034018393605947495
# epoch = 12, avg_loss = 0.028870202600955963
# epoch = 13, avg_loss = 0.02381684072315693
# epoch = 14, avg_loss = 0.019212961196899414
# epoch = 15, avg_loss = 0.015513855032622814
# Test Accuracy = 0.9786999821662903

# no activation
# epoch = 1, avg_loss = 0.3823098838329315
# epoch = 2, avg_loss = 0.314914345741272
# epoch = 3, avg_loss = 0.2996465861797333
# epoch = 4, avg_loss = 0.291761577129364
# epoch = 5, avg_loss = 0.2852623164653778
# epoch = 6, avg_loss = 0.2812996506690979
# epoch = 7, avg_loss = 0.2806396782398224
# epoch = 8, avg_loss = 0.279279887676239
# epoch = 9, avg_loss = 0.27399659156799316
# epoch = 10, avg_loss = 0.27329885959625244
# epoch = 11, avg_loss = 0.2690597176551819
# epoch = 12, avg_loss = 0.2677549421787262
# epoch = 13, avg_loss = 0.26621267199516296
# epoch = 14, avg_loss = 0.26553186774253845
# epoch = 15, avg_loss = 0.26366081833839417
# Test Accuracy = 0.9164000153541565
'''