# encoding=utf-8
import tensorflow as tf

INPUT_NODE=784
OUTPUT_NODE=10
IMAGE_CHANNELS=1
CONV1_SIZE=5
CONV1_DEEP=32
CONV2_SIZE=5
CONV2_DEEP=64
FC1_SIZE=512

def inference(input_tensor,train=False,regularizer=None):
    #layer1-conv1
    with tf.variable_scope("layer1-conv1"):
        kernel=tf.get_variable("kernel",[CONV1_SIZE,CONV1_SIZE,IMAGE_CHANNELS,CONV1_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias=tf.get_variable("bias",[CONV1_DEEP],initializer=tf.constant_initializer(0.0))
        conv1=tf.nn.conv2d(input_tensor,kernel,[1,1,1,1],padding='SAME')
        relu1=tf.nn.relu(tf.nn.bias_add(conv1,bias))
    #layer2-pool1
    with tf.variable_scope("layer2-pool1"):
        pool1=tf.nn.max_pool(relu1,[1,2,2,1],[1,2,2,1],padding='SAME')
    #layer3-conv2
    with tf.variable_scope("layer3-conv2"):
        kernel=tf.get_variable("kernel",[CONV2_SIZE,CONV2_SIZE,CONV1_DEEP,CONV2_DEEP],initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias=tf.get_variable("bias",[CONV2_DEEP],initializer=tf.constant_initializer(0.0))
        conv2=tf.nn.conv2d(pool1,kernel,[1,1,1,1],padding='SAME')
        relu2=tf.nn.relu(tf.nn.bias_add(conv2,bias))
    with tf.variable_scope("layer4-pool2"):
        pool2=tf.nn.max_pool(relu2,[1,2,2,1],[1,2,2,1],padding='SAME')

    pool2_shape=pool2.get_shape().as_list()
    nodes=pool2_shape[1]*pool2_shape[2]*pool2_shape[3]
    pool2_reshaped=tf.reshape(pool2,[pool2_shape[0],nodes])
    with tf.variable_scope('layer5-fc1'):
        weight=tf.get_variable("weight",[nodes,FC1_SIZE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        #只有全连接层的权重需要加入正则化
        if regularizer!=None:
            tf.add_to_collection("loss",regularizer(weight))
        bias=tf.get_variable("bias",[FC1_SIZE],initializer=tf.constant_initializer(0.1))
        fc1=tf.nn.relu(tf.matmul(pool2_reshaped,weight)+bias)
        #只有全连接层的节点使用drop_out，而且只有训练的时候需要dropout，测试不需要
        if train:
            fc1 = tf.nn.dropout(fc1, keep_prob=0.7)

    with tf.variable_scope("layer6-fc2"):
        weight=tf.get_variable("weight",[FC1_SIZE,OUTPUT_NODE],initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection("loss",regularizer(weight))
        bias=tf.get_variable("bias",[OUTPUT_NODE],initializer=tf.constant_initializer(0.1))
        logit=tf.matmul(fc1,weight)+bias

    return tf.nn.softmax(logit)