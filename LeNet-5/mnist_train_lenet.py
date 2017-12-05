# encoding=utf-8
"""
训练的模块，仅仅是训练和保存模型，没有评估
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference_lenet
import os
#学习率
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
#batch size
BATCH_SIZE=100
#正则化参数
REGULARIZATION_RATE=0.0001
#滑动平均衰减率
MOVING_AVERAGE_DECAY=0.99
#训练迭代次数
TRAINING_STEP=30000
#数据存放目录
DATA_PATH="/tmp/data"
#模型保存
MODEL_SAVE_PATH="/tmp/model/lenet-5"
MODEL_NAME="mnist.ckpt"

def train(mnist):
    x=tf.placeholder(tf.float32,[BATCH_SIZE,mnist_inference_lenet.IMAGE_SIZE,mnist_inference_lenet.IMAGE_SIZE,mnist_inference_lenet.NUM_CHANNELS],"x-input")
    y_=tf.placeholder(tf.float32,[BATCH_SIZE,mnist_inference_lenet.OUTPUT_NODE],"y-input")
    data_size=mnist.train.num_examples
    global_step=tf.Variable(0.0,dtype=tf.float32,trainable=False)

    #正则化项
    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y=mnist_inference_lenet.inference(x,True,regularizer)
    #交叉熵损失
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,axis=1),logits=y)
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=cross_entropy_mean+tf.add_n(tf.get_collection("loss"))
    #滑动平均值
    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,num_updates=global_step)
    ema_op=ema.apply(tf.trainable_variables())
    #学习率
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,data_size/BATCH_SIZE,LEARNING_RATE_DECAY)
    #保存模型
    saver=tf.train.Saver()

    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)
    with tf.control_dependencies([ema_op,train_step]):
        train_op=tf.no_op("train")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(TRAINING_STEP):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            xs_reshaped=tf.reshape(xs,[BATCH_SIZE,mnist_inference_lenet.IMAGE_SIZE,mnist_inference_lenet.IMAGE_SIZE,mnist_inference_lenet.NUM_CHANNELS])
            xs_reshaped=sess.run(xs_reshaped)
            _,loss_value,step=sess.run([train_op,loss,global_step],feed_dict={x:xs_reshaped,y_:ys})
            if i%1000 == 0:
                print("After %d step(s),loss on training batch is %f"%(step,loss_value))
                saver.save(sess,os.path.join(MODEL_SAVE_PATH,MODEL_NAME),global_step)

def main(argv=None):
    mnist=input_data.read_data_sets(DATA_PATH, one_hot=True)
    train(mnist)

if __name__ == '__main__':
    tf.app.run()