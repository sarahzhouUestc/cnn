# encoding=utf-8
"""实现inception-v3模型中的一个inception模块"""
import tensorflow as tf
import tensorflow.contrib.slim as slim

# 设置函数的默认参数取值
with slim.arg_scope([slim.conv2d,slim.max_pool2d,slim.avg_pool2d],stride=1,paddding='SAME'):
    net=[]
    #为一个inception模块声明一个统一的变量空间
    with tf.variable_scope("Mixed_7c"):
        with tf.variable_scope("Branch_0"):
            branch_0=slim.conv2d(net,320,[1,1],scope='Conv2d_0a_1x1')
        with tf.variable_scope("Branch_1"):
            branch_1=slim.conv2d(net,384,[1,1],scope='Conv2d_0a_1x1')
            #３表示矩阵是在深度这个维度上进行拼接
            branch_1=tf.concat(3,[slim.conv2d(branch_1,384,[1,3],scope='Conv2d_0b_1x3'),\
                                  slim.conv2d(branch_1,384,[3,1],scope='Conv2d_0c_3x1')])
        with tf.variable_scope("Branch_2"):
            branch_2=slim.conv2d(net,448,[1,1],scope='Conv2d_0a_1x1')
            branch_2=slim.conv2d(branch_2,384,[3,3],scope='Conv2d_0b_3x3')
            branch_2=tf.concat(3,[slim.conv2d(branch_2,384,[1,3],scope='Conv2d_0c_1x3'),\
                                  slim.conv2d(branch_2,384,[3,1],scope='Conv2d_0d_3x1')])
        with tf.variable_scope("Branch_3"):
            branch_3=slim.avg_pool2d(net,[3,3],scope='AvgPool_0a_3x3')
            branch_3=slim.conv2d(branch_3,192,[1,1],scope='Conv2d_0b_1x1')

        net=tf.concat(3,[branch_0,branch_1,branch_2,branch_3])