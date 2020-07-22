import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import glob
%matplotlib inline


class infogan():

    # to 28 * 28 * 1
    noise_dim = 50 
    class_dim = 10 # 0-9 十个数字
    batch_size = 256
    noise_cond_dim = 50

    def __init__(self):
        self.generator_optimizer = tf.keras.optimizers.Adam(1e-5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(1e-5)
        self.noise_seed = tf.random.normal([self.class_dim, self.noise_dim])
        self.cat_seed = np.random.randint(0, 10, size=(self.class_dim, 1))
        self.bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.cce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.generator = self.generator_model()
        self.discriminator = self.discriminator_model()



    def generator_model(self):
        noise_seed = layers.Input(shape=((self.noise_dim,)))
        cond_seed = layers.Input(shape=((self.noise_cond_dim,)))
        label = layers.Input(shape=(()))
        

        x = layers.Embedding(self.class_dim, self.noise_dim, input_length=1)(label)
        x = layers.Flatten()(x)
        x = layers.concatenate([noise_seed,cond_seed,x])
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

        model = keras.Model(inputs=[noise_seed,cond_seed,label],outputs=x)
        return model
    
    def generator_loss(self,fake_output,fake_pred,label,cond_out,cond_in):
        fake_loss = self.bce( tf.ones_like(fake_output),fake_output )
        fake_label = self.cce(label,fake_pred)
        cond_loss = tf.reduce_mean(tf.square(cond_out - cond_in))
        gen_loss = fake_loss + fake_label + cond_loss
        return gen_loss
    
    def discriminator_model(self):
        image = keras.layers.Input(shape=((28,28,1)))

        x = layers.Conv2D(32, (3,3), strides=(2,2), padding='same', use_bias=False)(image)
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
        x1 = layers.Dense(1)(x) #是否正确
        x2 = layers.Dense(self.class_dim)(x) #是数字几
        x3 = layers.Dense(self.noise_cond_dim, activation='sigmoid')(x) #输入的条件是否能恢复

        model = keras.Model(inputs=image, outputs=[x1,x2,x3])
        return model
    
    def discriminator_loss(self,real_output,fake_output,real_pred,label,cond_out,cond_in):
        real_loss = self.bce( tf.ones_like(real_output),real_output)
        fake_loss = self.bce( tf.zeros_like(fake_output),fake_output )
        real_label = self.cce( label, real_pred )
        cond_loss = tf.reduce_mean(tf.square(cond_out - cond_in))
        total_loss = real_loss + fake_loss + real_label + cond_loss
        return total_loss
    
    @tf.function
    def train_step(self,images,labels):
        batchsize = labels.shape[0]
        noise = tf.random.normal([batchsize, self.noise_dim])
        cond_in = tf.random.uniform((batchsize, self.noise_cond_dim))
        print(noise.shape,cond_in.shape,labels.shape )

        with tf.GradientTape() as gen_tape , tf.GradientTape() as disc_tape:
            generated_images = self.generator( (noise,cond_in,labels), training=True )

            real_output,real_pred,real_cond_out = self.discriminator( images ,training=True )
            fake_output,fake_pred,fake_cond_out = self.discriminator( generated_images ,training=True )

            gen_loss = self.generator_loss(fake_output,fake_pred,labels,fake_cond_out,cond_in)
            disc_loss = self.discriminator_loss(real_output,fake_output,real_pred,labels,fake_cond_out,cond_in)
        
        gradients_of_generator = gen_tape.gradient(gen_loss,self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(disc_loss,self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables ))
        self.discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator,self.discriminator.trainable_variables))
    

    def generate_images(self,epoch):
        print('Epoch:', epoch+1)
        cond_seed = tf.random.uniform((self.noise_seed.shape[0],self.noise_cond_dim))
        predictions = self.generator((self.noise_seed,cond_seed,self.cat_seed), training=False)
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
      filename = 'infogan.h5'
      self.generator.save(filename)
      self.generate_images(epoch)


if __name__ == "__main__":
  (train_image, train_label), ( _, _) = keras.datasets.mnist.load_data()
  train_image = train_image / 127.5  - 1
  train_image = np.expand_dims(train_image, -1)
  image_count = train_image.shape[0]
  dataset = tf.data.Dataset.from_tensor_slices((train_image, train_label))
  dataset = dataset.shuffle(image_count).batch(256)
  infogan = infogan()
  infogan.train(dataset,400)
