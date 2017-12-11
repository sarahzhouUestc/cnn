# encoding=utf-8
"""基于inception-v3的迁移学习"""
import tensorflow as tf
import glob
import numpy as np
import random
import os.path
from tensorflow.python.platform import gfile

INPUT_DATA="/tmp/data/flower_photos"
MODEL_FILE="/tmp/model/tensorflow_inception_graph.pb"
CACHE_DIR="/tmp/bottleneck"
BOTTLENECK_TENSOR_NAME="pool_3/_reshape:0"
BOTTLENECK_TENSOR_SIZE=2048
INPUT_IMAGE_TENSOR_NAME="DecodeJpeg/contents:0"
VALIDATION_PERCENTAGE=10
TEST_PERCENTAGE=10
EPOCHS=4000
BATCH=100
LEARNING_RATE=0.01

def create_image_lists():
    result={}
    sub_dirs=[x[0] for x in os.walk(INPUT_DATA)]
    is_root=True
    for dir in sub_dirs:
        if is_root:
            is_root=False
            continue
        #类别名称
        label_name=os.path.basename(dir)
        #获取当前类别的file_list，当前类别的所有file
        file_list=[]
        for ext in ['jpg','jpeg','JPG','JPEG']:
            file_path=os.path.join(INPUT_DATA,label_name,"*."+ext)
            file_list.extend(glob.glob(file_path))
        #划分training,validation,test
        training=[]
        validation=[]
        testing=[]
        for file in file_list:
            rdn=random.randrange(100)
            if rdn < VALIDATION_PERCENTAGE:
                validation.append(file)
            elif rdn < (VALIDATION_PERCENTAGE+TEST_PERCENTAGE):
                testing.append(file)
            else:
                training.append(file)
        result[label_name]={
            "dir":dir,
            "training":training,
            "validation":validation,
            "testing":testing
        }

    return result

#获取训练使用的bottlenecks向量
def create_or_get_random_bottlenecks(sess,image_lists,nums,classes,category,input_image_tensor,bottleneck_tensor):
    bottlenecks=[]
    groundtruths=[]
    for n in range(nums):
        label_index=random.randrange(classes)
        label_name=list(image_lists.keys())[label_index]
        groundtruth=[0 for x in range(classes)]
        groundtruth[label_index]=1

        category_file_list=image_lists[label_name][category]
        groundtruths.append(groundtruth)
        idx=random.randrange(65536)
        file_name=category_file_list[idx % len(category_file_list)]
        bottleneck_path=os.path.join(CACHE_DIR,label_name,file_name+".txt")
        bottlenecks.append(create_or_get_bottleneck(sess,bottleneck_path,label_name,input_image_tensor,bottleneck_tensor))
    return bottlenecks,groundtruths

def create_or_get_bottleneck(sess,path,label_name,input_image_tensor,bottleneck_tensor):
    if not os.path.exists(os.path.join(CACHE_DIR,label_name)):
        os.mkdir(os.path.join(CACHE_DIR,label_name))
    if os.path.exists(path):
        with open(path,'r') as bottleneck_file:
            bottleneck_string=bottleneck_file.read()
    else:
        image_path=get_image_path_by_bottleneck(path)
        image_data=gfile.FastGFile(image_path,mode='rb').read()
        bottleneck_values=sess.run(bottleneck_tensor,feed_dict={input_image_tensor:image_data})
        bottleneck_values=np.squeeze(bottleneck_values)
        bottleneck_string=','.join([str(x) for x in bottleneck_values])
        with open(path,'w') as b:
            b.write(bottleneck_string)
    bottleneck=[float(x) for x in bottleneck_string.split(',')]
    return bottleneck

def get_image_path_by_bottleneck(bottleneck_path):
    label_name=bottleneck_path.split('/')[-2]
    bottleneck_file_name=bottleneck_path.split('/')[-1]
    image_name=bottleneck_file_name.split(".")[-3]
    image_ext=bottleneck_file_name.split(".")[-2]
    return os.path.join(INPUT_DATA,label_name,image_name+"."+image_ext)

def get_test_bottlenecks(sess,image_lists,input_image_tensor,bottleneck_tensor):
    bottlenecks=[]
    groundtruths=[]
    for label_idx in range(len(image_lists.keys())):
        label_name=list(image_lists.keys())[label_idx]
        test=image_lists[label_name]['testing']
        for image_name in test:
            groundtruth=[0 for x in range(len(image_lists.keys()))]
            groundtruth[label_idx]=1
            groundtruths.append(groundtruth)
            path=os.path.join(CACHE_DIR,label_name,image_name+".txt")
            bottleneck=create_or_get_bottleneck(sess,path,label_name,input_image_tensor,bottleneck_tensor)
            bottlenecks.append(bottleneck)
    return bottlenecks,groundtruths

def main(argv=None):
    image_lists=create_image_lists();
    classes=len(image_lists.keys())

    #读取inception模型的protocol buffer文件
    with gfile.FastGFile(MODEL_FILE,mode="rb") as model:
        graph_def=tf.GraphDef()
        graph_def.ParseFromString(model.read())
        input_image_tensor,bottleneck_tensor=tf.import_graph_def(graph_def,return_elements=[INPUT_IMAGE_TENSOR_NAME,BOTTLENECK_TENSOR_NAME])

        #全连接层的placeholder定义
        x=tf.placeholder(tf.float32,[None,BOTTLENECK_TENSOR_SIZE],name="bottleneck-input")
        y_=tf.placeholder(tf.float32,[None,classes],name='groundtruth-input')
        weight=tf.Variable(tf.truncated_normal([BOTTLENECK_TENSOR_SIZE,classes],stddev=0.001),dtype=tf.float32,name='weight')
        bias=tf.Variable(tf.zeros([classes],dtype=tf.float32,name='bias'))
        y=tf.matmul(x,weight)+bias
        logit=tf.nn.softmax(y)
        #loss
        cross_entropy=tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.argmax(y_,1),logits=logit)
        loss=tf.reduce_mean(cross_entropy)
        train_step=tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)
        #accuracy
        correct_prediction=tf.equal(tf.argmax(y_,1),tf.argmax(logit,1))
        accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(EPOCHS):
                bottlenecks,ground_truths=create_or_get_random_bottlenecks(sess,image_lists,BATCH,classes,"training",input_image_tensor,bottleneck_tensor)
                sess.run(train_step,feed_dict={x:bottlenecks,y_:ground_truths})
                if i%100 == 0:
                    validation_bottlenecks,validation_groundtruths=create_or_get_random_bottlenecks(sess,image_lists,BATCH,classes,"validation",input_image_tensor,bottleneck_tensor)
                    acc=sess.run(accuracy,feed_dict={x:validation_bottlenecks,y_:validation_groundtruths})
                    print("After %d epochs, the accuracy on validation is %.1f%%"%(i,acc*100))
            test_bottlenecks,test_groundtruths=get_test_bottlenecks(sess,image_lists,input_image_tensor,bottleneck_tensor)
            test_acc=sess.run(accuracy,feed_dict={x:test_bottlenecks,y_:test_groundtruths})
            print("The accuracy on test is %.1f%%"%(test_acc*100))

if __name__ == '__main__':
    tf.app.run()