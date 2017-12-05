# encoding=utf-8
"""
用来测试和验证训练好的模型
"""
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
import mnist_inference_lenet
import mnist_train_lenet
#每10秒加载一次最新的模型，并在测试集上测试最新模型的正确率
EVAL_INTERNAL_SECS=10

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x=tf.placeholder(tf.float32,[mnist.validation.num_examples,mnist_inference_lenet.IMAGE_SIZE,mnist_inference_lenet.IMAGE_SIZE,mnist_inference_lenet.NUM_CHANNELS],name="x-input")
        y_=tf.placeholder(tf.float32,[mnist.validation.num_examples,mnist_inference_lenet.OUTPUT_NODE],name="y-input")
        #测试时不关心正则化损失的值，因为正则化损失是用来解决过拟合的
        y=mnist_inference_lenet.inference(x,False,None)
        correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

        #通过变量重命名的方式来加载模型
        ema=tf.train.ExponentialMovingAverage(mnist_train_lenet.MOVING_AVERAGE_DECAY)
        variables_to_restore=ema.variables_to_restore()     #字典
        saver=tf.train.Saver(variables_to_restore)
        while True:
            with tf.Session() as sess:
                xs=mnist.validation.images
                xs_reshaped = tf.reshape(xs, [mnist.validation.num_examples, mnist_inference_lenet.IMAGE_SIZE, mnist_inference_lenet.IMAGE_SIZE,
                                              mnist_inference_lenet.NUM_CHANNELS])
                xs_reshaped = sess.run(xs_reshaped)
                validate_feed={x:xs_reshaped, y_:mnist.validation.labels}
                ckpt=tf.train.get_checkpoint_state(mnist_train_lenet.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    #得到该模型保存时迭代的轮数
                    global_step=ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score=sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %f" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
                #每隔10秒加载一次
                time.sleep(EVAL_INTERNAL_SECS)

def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data", one_hot=True)
    evaluate(mnist)

if __name__ == '__main__':
    tf.app.run()
