import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

tf.random.set_seed(2019)

conv_layer=[
    layers.Conv2D(64, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
    layers.Conv2D(64, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    
    layers.Conv2D(128, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
    layers.Conv2D(128, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    
    layers.Conv2D(256, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
    layers.Conv2D(256, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    
    layers.Conv2D(512, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    
    layers.Conv2D(512, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
    layers.Conv2D(512, kernel_size=(3,3), padding='same', activation=tf.nn.relu),
    layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same'),
    
    layers.Flatten(),
    
    layers.Dense(4096, activation=tf.nn.relu),
    layers.Dense(4096, activation=tf.nn.relu),
    layers.Dense(100, activation=None)
]

model = Sequential(layers=conv_layer)
model.build(input_shape=[None, 32,32,3])
'''
跑模型的时候先给个随便什么输入，看看输出是不是期望的
'''
x=tf.random.normal([4,32,32,3])
out=model(x)
print(out.shape)

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.squeeze(y, axis=0)
    y = tf.cast(tf.one_hot(y, depth=100),dtype=tf.int32)
    return x, y

(x, y), (x_test, y_test) = datasets.cifar100.load_data()
print(y.shape)

train_db=tf.data.Dataset.from_tensor_slices((x,y)).shuffle(1000).map(preprocess).batch(128)
test_db=tf.data.Dataset.from_tensor_slices((x_test,y_test)).map(preprocess).batch(128)

optimizer = optimizers.Adam(lr=1e-4)
variables = model.trainable_variables
for epoch in range(50):

    for step, (x,y) in enumerate(train_db):

        with tf.GradientTape() as tape:
            # [b, 32, 32, 3] => [b, 1, 1, 512]
            logits = model(x)
            # compute loss
            loss = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
            loss = tf.reduce_mean(loss)

        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(zip(grads, variables))

        if step %100 == 0:
            print(epoch, step, 'loss:', float(loss))
    total_num = 0
    total_correct = 0
    for x,y in test_db:
        logits = model(x)
        prob = tf.nn.softmax(logits, axis=1)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)

        correct = tf.cast(tf.equal(pred, tf.cast(tf.argmax(y, axis=1), dtype=tf.int32)), dtype=tf.int32)
        correct = tf.reduce_sum(correct)

        total_num += x.shape[0]
        total_correct += int(correct)

    acc = total_correct / total_num
    print(epoch, 'acc:', acc)