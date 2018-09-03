import tensorflow as tf
tf.enable_eager_execution()
from PIL import Image
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import matplotlib.image as mpimg

mylist = os.listdir("./data/")
i = iter(mylist)

mylist2 = [os.path.join("data/", next(i)) for fname in mylist if os.path.isfile]
# print(mylist2)

# TODO:
# add input_shape() to 1st conv layers
# reshape images to 4D
# maybe make thowway labels


def _parse_function(filename):
    #image_string = tf.read_file(filename)
    image_string = tf.read_file(filename)
    decoded_image = tf.image.decode_png(image_string, channels=3)
    resize = tf.image.resize_images(decoded_image, [128, 128])
    return resize

# _ = tf.expand_dims(image_resized, axis=3)

# batch, channels, rows, cols input


# i = iter(mylist2)
# readarray = _parse_function(next(i))
# readarray = readarray.reshape(readarray.shape[2], 64, 64, 1)


dataset = tf.data.Dataset.from_tensor_slices(mylist2)
dataset = dataset.map(_parse_function)
Bdataset = dataset.batch(10, drop_remainder=False)


#iterator = Bdataset.make_one_shot_iterator()
#next_element = iterator.get_next()


# print(dataset)
# print(Bdataset)
# print(iterator)
# print(next_element)


class Generator(tf.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64 * 64 * 3, use_bias=False)
        self.batchnorm1 = tf.keras.layers.BatchNormalization()

        self.conv1 = tf.keras.layers.Conv2DTranspose(
            64, (5, 5), strides=(1, 1), padding='same', use_bias=False, input_shape=(128, 128, 3))
        self.batchnorm2 = tf.keras.layers.BatchNormalization()

        self.conv2 = tf.keras.layers.Conv2DTranspose(
            32, (5, 5), strides=(2, 2), padding='same', use_bias=False)
        self.batchnorm3 = tf.keras.layers.BatchNormalization()

        self.conv3 = tf.keras.layers.Conv2DTranspose(
            3, (5, 5), strides=(2, 2), padding='same', use_bias=False)

    def call(self, x, training=True):
        x = self.fc1(x)
        x = self.batchnorm1(x, training=training)
        x = tf.nn.relu(x)

        x = tf.reshape(x, shape=(-1, 32, 32, 3))

        x = self.conv1(x)
        x = self.batchnorm2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.batchnorm3(x, training=training)
        x = tf.nn.relu(x)

        x = tf.nn.tanh(self.conv3(x))
        return x


class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(64, (5, 5), strides=(
            2, 2), padding='same', input_shape=(128, 128, 3))
        self.conv2 = tf.keras.layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same')
        self.dropout = tf.keras.layers.Dropout(0.3)
        self.flatten = tf.keras.layers.Flatten()
        self.fc1 = tf.keras.layers.Dense(1)

    def call(self, x, training=True):
        x = tf.nn.leaky_relu(self.conv1(x))
        x = self.dropout(x, training=training)
        x = tf.nn.leaky_relu(self.conv2(x))
        x = self.dropout(x, training=training)
        #x = tf.reshape(x, shape=(-1, 4, 4, 512))
        x = self.flatten(x)
        x = self.fc1(x)
        return x


generator = Generator()
discriminator = Discriminator()


def discriminator_loss(real_output, generated_output):
    # [1,1,...,1] with real output since it is true and we want
    # our generated examples to look like it
    real_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.ones_like(real_output), logits=real_output)

    # [0,0,...,0] with generated images since they are fake
    generated_loss = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=tf.zeros_like(generated_output), logits=generated_output)

    total_loss = real_loss + generated_loss

    return total_loss


def generator_loss(generated_output):
    return tf.losses.sigmoid_cross_entropy(tf.ones_like(generated_output), generated_output)


discriminator_optimizer = tf.train.AdamOptimizer(1e-4)
generator_optimizer = tf.train.AdamOptimizer(1e-4)


checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)


EPOCHS = 150
noise_dim = 100
num_examples_to_generate = 15

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement of the gan.
latent_space = tf.random_normal([num_examples_to_generate,
                                 noise_dim])


def generate_and_save_images(model, epoch, test_input):
    # make sure the training parameter is set to False because we
    # don't want to train the batchnorm layer when doing inference.
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(10, 10))

    for i in range(predictions.shape[0]):
        i = 0
        plt.subplot(11, 11, i + 1)
        #plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5 / 255)
        plt.axis('off')

    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    # plt.show()


def train(dataset, epochs, noise_dim):
    for epoch in range(epochs):
        start = time.time()

        for images in dataset:
            # generating noise from a uniform distribution
            noise = tf.random_normal([10, noise_dim])

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                generated_images = generator(noise, training=True)

                real_output = discriminator(images, training=True)
                generated_output = discriminator(generated_images, training=True)

                gen_loss = generator_loss(generated_output)
                disc_loss = discriminator_loss(real_output, generated_output)

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

            generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.variables))
            discriminator_optimizer.apply_gradients(
                zip(gradients_of_discriminator, discriminator.variables))

        if epoch % 1 == 0:
            # plt.imshow(generated_output)
            generate_and_save_images(generator,
                                     epoch + 1,
                                     latent_space)

        # saving (checkpoint) the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for epoch {} is {} sec'.format(epoch + 1,
                                                         time.time() - start))
    # generating after the final epoch
    # plt.imshow(generated_output)
    generate_and_save_images(generator,
                             epochs,
                             latent_space)


train(Bdataset, EPOCHS, noise_dim)
