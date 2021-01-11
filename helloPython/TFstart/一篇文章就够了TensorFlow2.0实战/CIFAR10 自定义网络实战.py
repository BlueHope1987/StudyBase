import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
import os

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def preprocess(x, y):
    x=tf.cast(x, dtype=tf.float32)/255.
    y=tf.squeeze(y)  # y的原始shape是(5000,1) 第二维是多余的
    y=tf.one_hot(y, depth=10)
    y=tf.cast(y, dtype=tf.int32)
    return x, y

batch_size=128
(x, y), (x_val, y_val)=datasets.cifar10.load_data()

train_db=tf.data.Dataset.from_tensor_slices((x,y))
train_db=train_db.map(preprocess).shuffle(10000).batch(batch_size)

val_db=tf.data.Dataset.from_tensor_slices((x_val,y_val))
val_db=val_db.map(preprocess).shuffle(10000).batch(batch_size)

class MyDense(layers.Layer):
    def __init__(self, in_dim, out_dim):
        super(MyDense, self).__init__()
        self.kernel=self.add_variable('w', [in_dim, out_dim])
        
    def call(self, inputs, training=None):
        x=inputs@self.kernel
        return x
 
class MyNet(keras.Model):
    def __init__(self):
        super(MyNet, self).__init__()
        self.fc1=MyDense(32*32*3, 256)
        self.fc2=MyDense(256,128)
        self.fc3=MyDense(128,64)
        self.fc4=MyDense(64,32)
        self.fc5=MyDense(32,10)
    
    def call(self, inputs, training=None):
        x=tf.reshape(inputs, [-1, 32*32*3])
        x=self.fc1(x)
        x=tf.nn.relu(x)
        x=self.fc2(x)
        x=tf.nn.relu(x)
        x=self.fc3(x)
        x=tf.nn.relu(x)
        x=self.fc4(x)
        x=tf.nn.relu(x)
        x=self.fc5(x)
        
        return x

model = MyNet()
model.compile(
    optimizer=optimizers.Adam(), 
    loss=tf.losses.CategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)
model.fit(train_db, epochs=10, validation_data=val_db)