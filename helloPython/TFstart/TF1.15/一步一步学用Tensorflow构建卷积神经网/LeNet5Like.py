import tensorflow as tf
import numpy as np
from mnist import MNIST
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


mnist_folder = './helloPython/data/mnist/' #修改 项目根目录相对路径
mnist_image_width = 28
mnist_image_height = 28
mnist_image_depth = 1
mnist_num_labels = 10
mnist_image_size = 28 #debuging 后加
mndata = MNIST(mnist_folder)
mnist_train_dataset_, mnist_train_labels_ = mndata.load_training()
mnist_test_dataset_, mnist_test_labels_ = mndata.load_testing()
mnist_train_dataset, mnist_train_labels = reformat_data(mnist_train_dataset_, mnist_train_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)
mnist_test_dataset, mnist_test_labels = reformat_data(mnist_test_dataset_, mnist_test_labels_, mnist_image_size, mnist_image_size, mnist_image_depth)
print("There are {} images, each of size {}".format(len(mnist_train_dataset), len(mnist_train_dataset[0])))
print("Meaning each image has the size of 28*28*1 = {}".format(mnist_image_size*mnist_image_size*1))
print("The training set contains the following {} labels: {}".format(len(np.unique(mnist_train_labels_)), np.unique(mnist_train_labels_)))
print('Training set shape', mnist_train_dataset.shape, mnist_train_labels.shape)
print('Test set shape', mnist_test_dataset.shape, mnist_test_labels.shape)
train_dataset_mnist, train_labels_mnist = mnist_train_dataset, mnist_train_labels
test_dataset_mnist, test_labels_mnist = mnist_test_dataset, mnist_test_labels
######################################################################################
cifar10_folder = './helloPython/data/cifar10/' #修改 项目根目录相对路径
train_datasets = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', ]
test_dataset = ['test_batch']
c10_image_height = 32
c10_image_width = 32
c10_image_depth = 3
c10_num_labels = 10
c10_image_size=32 #debuging 后加
with open(cifar10_folder + test_dataset[0], 'rb') as f0:
    c10_test_dict = pickle.load(f0, encoding='bytes') #import pickle
c10_test_dataset, c10_test_labels = c10_test_dict[b'data'], c10_test_dict[b'labels']
test_dataset_cifar10, test_labels_cifar10 = reformat_data(c10_test_dataset, c10_test_labels, c10_image_size, c10_image_size, c10_image_depth)
c10_train_dataset, c10_train_labels = [], []
for train_dataset in train_datasets:
    with open(cifar10_folder + train_dataset, 'rb') as f0:
        c10_train_dict = pickle.load(f0, encoding='bytes')
        c10_train_dataset_, c10_train_labels_ = c10_train_dict[b'data'], c10_train_dict[b'labels']
        c10_train_dataset.append(c10_train_dataset_)
        c10_train_labels += c10_train_labels_
c10_train_dataset = np.concatenate(c10_train_dataset, axis=0)
train_dataset_cifar10, train_labels_cifar10 = reformat_data(c10_train_dataset, c10_train_labels, c10_image_size, c10_image_size, c10_image_depth)
del c10_train_dataset
del c10_train_labels
print("The training set contains the following labels: {}".format(np.unique(c10_train_dict[b'labels'])))
print('Training set shape', train_dataset_cifar10.shape, train_labels_cifar10.shape)
print('Test set shape', test_dataset_cifar10.shape, test_labels_cifar10.shape)

image_width = mnist_image_width
image_height = mnist_image_height
image_depth = mnist_image_depth
num_labels = mnist_num_labels
batch_size=32 ##debuging 后加 32->
#the dataset
train_dataset = mnist_train_dataset
train_labels = mnist_train_labels
test_dataset = mnist_test_dataset
test_labels = mnist_test_labels
#number of iterations and learning rate
num_steps = 50 #debug 10001 -> 
display_step = 1 #debug 1000 ->
learning_rate = 0.01 #debug 0.5->

#=========以上为复用代码==============
#=========以下为示范代码==============



LENET5_LIKE_BATCH_SIZE = 32
LENET5_LIKE_FILTER_SIZE = 5
LENET5_LIKE_FILTER_DEPTH = 16
LENET5_LIKE_NUM_HIDDEN = 120
 
def variables_lenet5_like(filter_size = LENET5_LIKE_FILTER_SIZE, 
                          filter_depth = LENET5_LIKE_FILTER_DEPTH, 
                          num_hidden = LENET5_LIKE_NUM_HIDDEN,
                          image_width = 28, image_depth = 1, num_labels = 10):
 
    w1 = tf.Variable(tf.truncated_normal([filter_size, filter_size, image_depth, filter_depth], stddev=0.1))
    b1 = tf.Variable(tf.zeros([filter_depth]))
 
    w2 = tf.Variable(tf.truncated_normal([filter_size, filter_size, filter_depth, filter_depth], stddev=0.1))
    b2 = tf.Variable(tf.constant(1.0, shape=[filter_depth]))
 
    w3 = tf.Variable(tf.truncated_normal([(image_width // 4)*(image_width // 4)*filter_depth , num_hidden], stddev=0.1))
    b3 = tf.Variable(tf.constant(1.0, shape = [num_hidden]))
 
    w4 = tf.Variable(tf.truncated_normal([num_hidden, num_hidden], stddev=0.1))
    b4 = tf.Variable(tf.constant(1.0, shape = [num_hidden]))
 
    w5 = tf.Variable(tf.truncated_normal([num_hidden, num_labels], stddev=0.1))
    b5 = tf.Variable(tf.constant(1.0, shape = [num_labels]))
    variables = {
                  'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
                  'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5
                }
    return variables
 
def model_lenet5_like(data, variables):
    layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='SAME')
    layer1_actv = tf.nn.relu(layer1_conv + variables['b1'])
    layer1_pool = tf.nn.avg_pool(layer1_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='SAME')
    layer2_actv = tf.nn.relu(layer2_conv + variables['b2'])
    layer2_pool = tf.nn.avg_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
 
    flat_layer = flatten_tf_array(layer2_pool)
    layer3_fccd = tf.matmul(flat_layer, variables['w3']) + variables['b3']
    layer3_actv = tf.nn.relu(layer3_fccd)
    #layer3_drop = tf.nn.dropout(layer3_actv, 0.5)
 
    layer4_fccd = tf.matmul(layer3_actv, variables['w4']) + variables['b4']
    layer4_actv = tf.nn.relu(layer4_fccd)
   #layer4_drop = tf.nn.dropout(layer4_actv, 0.5)
 
    logits = tf.matmul(layer4_actv, variables['w5']) + variables['b5']
    return logits



#parameters determining the model size
image_size = mnist_image_size
num_labels = mnist_num_labels
 
#the datasets
train_dataset = mnist_train_dataset
train_labels = mnist_train_labels
test_dataset = mnist_test_dataset
test_labels = mnist_test_labels
 
#number of iterations and learning rate
num_steps = 10001
display_step = 1  #DEBUG 1000->
learning_rate = 0.001
 
graph = tf.Graph()
with graph.as_default():
    #1) First we put the input data in a Tensorflow friendly form. 
    tf_train_dataset = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
    tf_train_labels = tf.compat.v1.placeholder(tf.float32, shape = (batch_size, num_labels))
    tf_test_dataset = tf.compat.v1.constant(test_dataset, tf.float32)
 
    #2) Then, the weight matrices and bias vectors are initialized
    variables = variables_lenet5_like(image_depth = image_depth, num_labels = num_labels)
 
    #3. The model used to calculate the logits (predicted labels)
    model = model_lenet5_like
    logits = model(tf_train_dataset, variables)
 
    #4. then we compute the softmax cross entropy between the logits and the (actual) labels
    loss = tf.compat.v1.reduce_mean(tf.compat.v1.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))
    
    #5. The optimizer is used to calculate the gradients of the loss function 
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(loss)
 
    # Predictions for the training, validation, and test data.
    train_prediction = tf.compat.v1.nn.softmax(logits)
    test_prediction = tf.compat.v1.nn.softmax(model(tf_test_dataset, variables))




with tf.compat.v1.Session(graph=graph) as session:
    tf.compat.v1.global_variables_initializer().run()
    print('Initialized with learning_rate', learning_rate)
    for step in range(num_steps):
 
        #Since we are using stochastic gradient descent, we are selecting  small batches from the training dataset,
        #and training the convolutional neural network each time with a batch. 
        offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
        batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
        batch_labels = train_labels[offset:(offset + batch_size), :]
        feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
        _, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict) #DEBUG: 停滞？
        
        if step % display_step == 0:
            train_accuracy = accuracy(predictions, batch_labels)
            test_accuracy = accuracy(test_prediction.eval(), test_labels)
            message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(step, l, train_accuracy, test_accuracy)
            print(message)