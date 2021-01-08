import tensorflow as tf
import numpy as np
import pickle


def randomize(dataset, labels):
    permutation = np.random.permutation(labels.shape[0])
    shuffled_dataset = dataset[permutation, :, :]
    shuffled_labels = labels[permutation]
    return shuffled_dataset, shuffled_labels
def one_hot_encode(np_array):
    return (np.arange(10) == np_array[:,None]).astype(np.float32)
def reformat_data(dataset, labels, image_width, image_height, image_depth):
    np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
    np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
    np_dataset, np_labels = randomize(np_dataset_, np_labels_)
    return np_dataset, np_labels
def flatten_tf_array(array):
    shape = array.get_shape().as_list()
    return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])
def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


#=========以上为复用代码==============
#=========以下为示范代码==============
from utils import *

#Code 3.1.1

ox17_image_width = 224
ox17_image_height = 224
ox17_image_depth = 3
ox17_num_labels = 17
 
import tflearn.datasets.oxflower17 as oxflower17
train_dataset_, train_labels_ = oxflower17.load_data(one_hot=True)
train_dataset_ox17, train_labels_ox17 = train_dataset_[:1000,:,:,:], train_labels_[:1000,:]
test_dataset_ox17, test_labels_ox17 = train_dataset_[1000:,:,:,:], train_labels_[1000:,:]
 
print('Training set', train_dataset_ox17.shape, train_labels_ox17.shape)
print('Test set', test_dataset_ox17.shape, test_labels_ox17.shape)


#Code 3.1.2

ALEX_PATCH_DEPTH_1, ALEX_PATCH_DEPTH_2, ALEX_PATCH_DEPTH_3, ALEX_PATCH_DEPTH_4 = 96, 256, 384, 256
ALEX_PATCH_SIZE_1, ALEX_PATCH_SIZE_2, ALEX_PATCH_SIZE_3, ALEX_PATCH_SIZE_4 = 11, 5, 3, 3
ALEX_NUM_HIDDEN_1, ALEX_NUM_HIDDEN_2 = 4096, 4096
 
 
def variables_alexnet(patch_size1 = ALEX_PATCH_SIZE_1, patch_size2 = ALEX_PATCH_SIZE_2, 
                      patch_size3 = ALEX_PATCH_SIZE_3, patch_size4 = ALEX_PATCH_SIZE_4, 
                      patch_depth1 = ALEX_PATCH_DEPTH_1, patch_depth2 = ALEX_PATCH_DEPTH_2, 
                      patch_depth3 = ALEX_PATCH_DEPTH_3, patch_depth4 = ALEX_PATCH_DEPTH_4, 
                      num_hidden1 = ALEX_NUM_HIDDEN_1, num_hidden2 = ALEX_NUM_HIDDEN_2,
                      image_width = 224, image_height = 224, image_depth = 3, num_labels = 17):
 
    w1 = tf.Variable(tf.truncated_normal([patch_size1, patch_size1, image_depth, patch_depth1], stddev=0.1))
    b1 = tf.Variable(tf.zeros([patch_depth1]))
 
    w2 = tf.Variable(tf.truncated_normal([patch_size2, patch_size2, patch_depth1, patch_depth2], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[patch_depth2]))
 
    w3 = tf.Variable(tf.truncated_normal([patch_size3, patch_size3, patch_depth2, patch_depth3], stddev=0.1))
    b3 = tf.Variable(tf.zeros([patch_depth3]))
 
    w4 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth3, patch_depth3], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape=[patch_depth3]))
 
    w5 = tf.Variable(tf.truncated_normal([patch_size4, patch_size4, patch_depth3, patch_depth3], stddev=0.1))
    b5 = tf.Variable(tf.zeros([patch_depth3]))
 
    pool_reductions = 3
    conv_reductions = 2
    no_reductions = pool_reductions + conv_reductions
    w6 = tf.Variable(tf.truncated_normal([(image_width // 2**no_reductions)*(image_height // 2**no_reductions)*patch_depth3, num_hidden1], stddev=0.1))
    b6 = tf.Variable(tf.constant(1.0, shape = [num_hidden1]))
 
    w7 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
    b7 = tf.Variable(tf.constant(1.0, shape = [num_hidden2]))
 
    w8 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
    b8 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
 
    variables = {
                 'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5, 'w6': w6, 'w7': w7, 'w8': w8, 
                 'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5, 'b6': b6, 'b7': b7, 'b8': b8
                }
    return variables
 
 
def model_alexnet(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 4, 4, 1], padding='SAME')
    layer1_relu = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.max_pool(layer1_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer1_norm = tf.nn.local_response_normalization(layer1_pool)
 
    layer2_conv = tf.nn.conv2d(layer1_norm, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_relu = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.max_pool(layer2_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer2_norm = tf.nn.local_response_normalization(layer2_pool)
 
    layer3_conv = tf.nn.conv2d(layer2_norm, variables['w3'], [1, 1, 1, 1], padding='SAME')
    layer3_relu = tf.nn.relu(layer3_conv + variables['b3'])
 
    layer4_conv = tf.nn.conv2d(layer3_relu, variables['w4'], [1, 1, 1, 1], padding='SAME')
    layer4_relu = tf.nn.relu(layer4_conv + variables['b4'])
 
    layer5_conv = tf.nn.conv2d(layer4_relu, variables['w5'], [1, 1, 1, 1], padding='SAME')
    layer5_relu = tf.nn.relu(layer5_conv + variables['b5'])
    layer5_pool = tf.nn.max_pool(layer4_relu, [1, 3, 3, 1], [1, 2, 2, 1], padding='SAME')
    layer5_norm = tf.nn.local_response_normalization(layer5_pool)
 
    flat_layer = flatten_tf_array(layer5_norm)
    layer6_fccd = tf.matmul(flat_layer, variables['w6']) + variables['b6']
    layer6_tanh = tf.tanh(layer6_fccd)
    layer6_drop = tf.nn.dropout(layer6_tanh, 0.5)
 
    layer7_fccd = tf.matmul(layer6_drop, variables['w7']) + variables['b7']
    layer7_tanh = tf.tanh(layer7_fccd)
    layer7_drop = tf.nn.dropout(layer7_tanh, 0.5)
 
    logits = tf.matmul(layer7_drop, variables['w8']) + variables['b8']
    return logits

#自行补全运行逻辑