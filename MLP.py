'''
Copyright (C) 2019 May 26 By JSH all rights reserved
Written by Sanghyeon Jo <josanghyeokn@gmail.com>
'''

import tensorflow as tf
from Define import *

'''
https://www.tensorflow.org/api_docs/python/tf/layers/dense

tf.layers.dense(
    inputs,
    units,
    activation=None,
    use_bias=True,
    kernel_initializer=None,
    bias_initializer=tf.zeros_initializer(),
    kernel_regularizer=None,
    bias_regularizer=None,
    activity_regularizer=None,
    kernel_constraint=None,
    bias_constraint=None,
    trainable=True,
    name=None,
    reuse=None
)
'''
def MLP(x, activation = False):
    x # [?, 784]

    x = tf.layers.dense(inputs = x, units = 256, name = 'fc_1') #[?, 256]
    if activation:
        x = tf.nn.sigmoid(x)

    x = tf.layers.dense(inputs = x, units = 256, name = 'fc_2') #[?, 256]
    if activation:
        x = tf.nn.sigmoid(x)
    
    logits = tf.layers.dense(inputs = x, units = CLASSES, name = 'fc_3') #[?, CLASSES]
    
    return logits

if __name__ == '__main__':
    input_var = tf.placeholder(tf.float32, [None, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_CHANNEL])

    logits = MLP(input_var)
    print(logits)

'''
Tensor("fc_3/BiasAdd:0", shape=(?, 10), dtype=float32)
'''