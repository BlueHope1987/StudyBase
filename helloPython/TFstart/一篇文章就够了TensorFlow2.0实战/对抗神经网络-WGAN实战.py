class Generator(keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        # z:[b, 100] => [b, 3*3*512] => [b, 3,3,512] =>[b,64,64,3]
        self.fc= layers.Dense(3*3*512)
        self.conv1=layers.Conv2DTranspose(256, 3, 3)
        self.bn1 = BatchNormalization()
        
        self.conv2=layers.Conv2DTranspose(128, 5, 2)
        self.bn2 = BatchNormalization()
        
        self.conv3=layers.Conv2DTranspose(3, 4, 3)
        
        
        
    def call(self, inputs, training=None):
        x = self.fc(inputs)
        x = tf.reshape(x, [-1, 3, 3, 512])
        x = tf.nn.leaky_relu(x)
        x = tf.nn.leaky_relu(self.bn1(self.conv1(x), training=training))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = self.conv3(x)
        x=tf.tanh(x) # [-1, 1]  Discriminator的输入是[-1,1]
        return x

class Discriminator(keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        # [b, 64, 64, 3] => [b,1]
        self.conv1 = layers.Conv2D(filters=64, kernel_size=5, strides=3, padding='valid')
        self.conv2 = layers.Conv2D(128, 5, 3)
        self.bn2 = BatchNormalization()
        self.conv3 = layers.Conv2D(256, 5, 3)
        self.bn3 = BatchNormalization()
        # [b, h, w, 3] => [b, -1]
        self.flatten = layers.Flatten()
        self.fc = layers.Dense(1)
        
    def call(self, inputs, training=None):
        x = tf.nn.leaky_relu(self.conv1(inputs))
        x = tf.nn.leaky_relu(self.bn2(self.conv2(x), training=training))
        x = tf.nn.leaky_relu(self.bn3(self.conv3(x), training=training))
        x= self.flatten(x)
        logits = self.fc(x)
        return logits
    
def celoss_ones(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.ones_like(logits))
    return tf.reduce_mean(loss)


def celoss_zeros(logits):
    # [b, 1]
    # [b] = [1, 1, 1, 1,]
    loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,
                                                   labels=tf.zeros_like(logits))
    return tf.reduce_mean(loss)

def gradient_penalty(discriminator, batch_x, fake_image):
    batchsz = batch_x.shape[0]
    #[b, h, w, c]
    t=tf.random.uniform([batchsz, 1, 1, 1])
    t = tf.broadcast_to(t, batch_x.shape)
    interplate = t*batch_x+(1-t)*fake_image
    with tf.GradientTape() as tape:
        tape.watch([interplate])
        d_interplote_logits = discriminator(interplate)
    grads = tape.gradient(d_interplote_logits, interplate)

    # grads:[b, h, w, c] => [b, -1]
    grads = tf.reshape(grads, [grads.shape[0], -1])
    gp = tf.norm(grads, axis=1) #[b]
    gp = tf.reduce_mean( (gp-1)**2 )

    return gp

def d_loss_fn(generator, discriminator, batch_z, batch_x, is_training):
    # 1. treat real image as real
    # 2. treat generated image as fake
    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    d_real_logits = discriminator(batch_x, is_training)

    d_loss_real = celoss_ones(d_real_logits)
    d_loss_fake = celoss_zeros(d_fake_logits)
    gp = gradient_penalty(discriminator, batch_x, fake_image)
    loss = d_loss_fake + d_loss_real + 5. * gp

    return loss, gp


def g_loss_fn(generator, discriminator, batch_z, is_training):

    fake_image = generator(batch_z, is_training)
    d_fake_logits = discriminator(fake_image, is_training)
    loss = celoss_ones(d_fake_logits)

    return loss

z_dim = 100
epochs = 3000000
batch_size = 512
learning_rate = 0.002
is_training = True

generator = Generator()
generator.build(input_shape = (None, z_dim))
discriminator = Discriminator()
discriminator.build(input_shape=(None, 64, 64, 3))
g_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
d_optimizer = tf.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)

for epoch in range(epochs):

    batch_z = tf.random.uniform([batch_size, z_dim], minval=-1., maxval=1.)
    batch_x = next(db_iter)

    # train D
    with tf.GradientTape() as tape:
        d_loss, gp = d_loss_fn(generator, discriminator, batch_z, batch_x, is_training)
    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    d_optimizer.apply_gradients(zip(grads, discriminator.trainable_variables))


    with tf.GradientTape() as tape:
        g_loss = g_loss_fn(generator, discriminator, batch_z, is_training)
    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_optimizer.apply_gradients(zip(grads, generator.trainable_variables))

    if epoch % 100 == 0:
        print(epoch, 'd-loss:',float(d_loss), 'g-loss:', float(g_loss),
                  'gp:', float(gp))

        z = tf.random.uniform([100, z_dim])
        fake_image = generator(z, training=False)
        img_path = os.path.join('images', 'wgan-%d.png'%epoch)
        save_result(fake_image.numpy(), 10, img_path, color_mode='P') 