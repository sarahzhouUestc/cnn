# encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import inference
import train
import time

EVAL_INTERVAL_SECS=10

def eval(mnist):
    x=tf.placeholder(tf.float32,[mnist.validation.num_examples,train.IMAGE_SIZE,train.IMAGE_SIZE,inference.IMAGE_CHANNELS],'x-input')
    y_=tf.placeholder(tf.float32,[mnist.validation.num_examples,inference.OUTPUT_NODE],'y-input')
    y=inference.inference(x,False,None)
    accuracy=tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y,1),tf.argmax(y_,1)),tf.float32))
    saver=tf.train.Saver(tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY).variables_to_restore(),max_to_keep=10)
    with tf.Session() as sess:
        xs=mnist.validation.images
        ys=mnist.validation.labels
        xs_reshaped=tf.reshape(xs,[mnist.validation.num_examples,train.IMAGE_SIZE,train.IMAGE_SIZE,inference.IMAGE_CHANNELS])
        validation_feed={x:sess.run(xs_reshaped),y_:ys}
        while True:
            ckpt=tf.train.get_checkpoint_state("/tmp/model/mylenet-5","checkpoint")
            if ckpt and ckpt.model_checkpoint_path:
                save_path=ckpt.model_checkpoint_path
                saver.restore(sess,save_path)
                acc=sess.run(accuracy,validation_feed)
                step=save_path.split("/")[-1].split("-")[-1]
                print("After {} steps, the accuracy on validation is {}".format(step,acc))
                time.sleep(EVAL_INTERVAL_SECS)

def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
    eval(mnist)

if __name__ == '__main__':
    tf.app.run()