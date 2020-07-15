import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import glob
%matplotlib inline


class cgan():

    # to 28 * 28 * 1
    noise_dim = 50 
    class_dim = 10 # 0-9 十个数字
    batch_size = 256

    def __init__(self):
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
        self.noise_seed = tf.random.normal([self.class_dim, self.noise_dim])
        self.cat_seed = np.random.randint(0, 10, size=(self.class_dim, 1))
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()



    def generator_model(self):
        seed = layers.Input(shape=((self.noise_dim,)))
        label = layers.Input(shape=(()))

        x = layers.Embedding(self.class_dim, self.noise_dim, input_length=1)(label)
        x = layers.Flatten()(x)
        x = layers.concatenate([seed, x])
        x = layers.Dense(3*3*128, use_bias=False)(x)
        x = layers.Reshape((3,3,128))(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)

        x = layers.Conv2DTranspose(64, (3,3), strides=(2,2), use_bias=False )(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x) #7*7

        x = layers.Conv2DTranspose(32, (3,3), strides=(2,2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x) #14*14

        x = layers.Conv2DTranspose(1, (3,3), strides=(2,2), padding='same', use_bias=False)(x)
        x = layers.Activation('tanh')(x) #28*28

        model = keras.Model(inputs=[seed,label],outputs=x)
        return model
    
    def generator_loss(self,fake_output):
        fake_loss = self.bce( tf.ones_like(fake_output),fake_output )
        return fake_loss
    
    def discriminator_model(self):
        image = keras.layers.Input(shape=((28,28,1)))
        label = keras.layers.Input(shape=(()))

        x = layers.Embedding(self.class_dim, 28*28, input_length=1)(label)
        x = layers.Reshape((28,28,1))(x)
        x = layers.concatenate([image, x])

        x = layers.Conv2D(32, (3,3), strides=(2,2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x) #14*14
        x = layers.Dropout(0.5)(x)
        
        x = layers.Conv2D(32 * 2, (3,3), strides=(2,2), padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x) #7*7
        x = layers.Dropout(0.5)(x)

        x = layers.Conv2D(32 * 2 * 2, (3,3), strides=(2,2), padding='same', use_bias=False )(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU()(x) #7*7
        x = layers.Dropout(0.5)(x)

        x = layers.Flatten()(x)
        x1 = layers.Dense(1)(x)

        model = keras.Model(inputs=[image,label], outputs=x1)
        return model
    
    def discriminator_loss(self,real_output,fake_output):
        real_loss = self.bce( tf.ones_like(real_output),real_output)
        fake_loss = self.bce( tf.zeros_like(fake_output),fake_output )
        total_loss = real_loss + fake_loss
        return total_loss
    
    @tf.function
    def train_step(self,images,labels):
        batchsize = labels.shape[0]
        noise = tf.random.normal([batchsize, self.noise_dim])

        with tf.GradientTape() as gen_tape , tf.GradientTape() as disc_tape:
            generated_images = self.generator( (noise,labels), training=True )

            real_output = self.discriminator( (images,labels),training=True )
            fake_output = self.discriminator( (generated_images,labels),training=True )

            gen_loss = self.generator_loss(fake_output)
            disc_loss = self.discriminator_loss(real_output,fake_output)
        
        gradients_of_generator = gen_tape.gradient(gen_loss,self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss,self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables ))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,self.discriminator.trainable_variables))
    

    def generate_images(self,epoch):
        print('Epoch:', epoch+1)
        predictions = self.generator((self.noise_seed, self.cat_seed), training=False)
        predictions = tf.squeeze(predictions)
        fig = plt.figure(figsize=(10, 1))

        for i in range(predictions.shape[0]):
            plt.subplot(1, 10, i+1)
            plt.imshow((predictions[i, :, :] + 1)/2 , cmap='gray'  )
            plt.axis('off')
        plt.show()

    def train(self,dataset,epochs):
      print(self.cat_seed.T)
      for epoch in range(epochs):
          for image_batch, label_batch in dataset:
              self.train_step(image_batch, label_batch)

          if epoch%10 == 0:
              self.generate_images(epoch)
      
      self.generate_images(epoch)

if __name__ == "__main__":
  (train_image, train_label), ( _, _) = keras.datasets.mnist.load_data()
  train_image = train_image / 127.5  - 1
  train_image = np.expand_dims(train_image, -1)
  image_count = train_image.shape[0]
  dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label))
  dataset = dataset.shuffle(image_count).batch(256)
  cgan = cgan()
  cgan.train(dataset,400)
