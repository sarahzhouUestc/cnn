# encoding=utf-8
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import inference

IMAGE_SIZE=28
IMAGE_CHANNELS=1
BATCH_SIZE=100
REGULARIZATION_RATE=0.0001
LEARNING_RATE_BASE=0.8
LEARNING_RATE_DECAY=0.99
MOVING_AVERAGE_DECAY=0.99
EPOCHS=5000

def train(mnist):
    x=tf.placeholder(tf.float32,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNELS],name='x-input')
    y_=tf.placeholder(tf.float32,[BATCH_SIZE,inference.OUTPUT_NODE],name='y-input')
    global_step=tf.Variable(0,dtype=tf.int32,trainable=False)
    data_size=mnist.train.num_examples

    regularizer=tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y=inference.inference(x,True,regularizer)
    cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y,labels=tf.argmax(y_,1))
    cross_entropy_mean=tf.reduce_mean(cross_entropy)
    loss=tf.add_n(tf.get_collection("loss"))+cross_entropy_mean
    #learning_rate=learning_rate_base*learning_rate_decay^(global_step/decay_steps)
    learning_rate=tf.train.exponential_decay(LEARNING_RATE_BASE,global_step,data_size/BATCH_SIZE,LEARNING_RATE_DECAY)
    train_step=tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step)
    ema=tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY,num_updates=global_step)
    ema_op=ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step,ema_op]):
        train_op=tf.no_op("train")
    correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(y_,1))
    accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    saver=tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(EPOCHS):
            xs,ys=mnist.train.next_batch(BATCH_SIZE)
            xs_reshaped=tf.reshape(xs,[BATCH_SIZE,IMAGE_SIZE,IMAGE_SIZE,IMAGE_CHANNELS])
            sess.run(train_op,feed_dict={x:sess.run(xs_reshaped),y_:ys})
            if i%100==0:
                acc=sess.run(accuracy,feed_dict={x:sess.run(xs_reshaped),y_:ys})
                print("After {} step(s), the accuracy on the training batch is {}".format(i,acc))
                saver.save(sess,"/tmp/model/mylenet-5/mnist.ckpt",global_step)

def main(argv=None):
    mnist=input_data.read_data_sets("/tmp/data",one_hot=True)
    train(mnist)
if __name__ == '__main__':
    tf.app.run()