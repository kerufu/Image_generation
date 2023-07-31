import time

import tensorflow
import numpy
import cv2

from myP.settings import STATICFILES_DIRS
from learningModel.setting import batchSize, imageSize, featureVectorLength, StandardGANPath
from learningModel.datasetManager import data_manager

class Generator(tensorflow.keras.Model):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = [
            tensorflow.keras.layers.Dense(imageSize*imageSize, activation='selu'),
            tensorflow.keras.layers.Reshape((imageSize//4, imageSize//4, 16)),
            tensorflow.keras.layers.Conv2DTranspose(8, 2, strides=2, activation='selu'),
            tensorflow.keras.layers.Conv2DTranspose(3, 2, strides=2)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class Discriminator(tensorflow.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = [
            tensorflow.keras.layers.Conv2D(8, 2, activation='selu'),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(16, activation='selu'),
            tensorflow.keras.layers.Dense(1)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class StandardGANWoker():
    GPath, DPath = StandardGANPath

    def __init__(self):
        self.G = Generator()
        self.D = Discriminator()
        try:
            self.G.load_weights(self.GPath)
            self.D.load_weights(self.DPath)
        except:
            pass
        self.trainDataReady = False
        self.cross_entropy = tensorflow.keras.losses.BinaryFocalCrossentropy(
            from_logits=True)
        self.GOptimizer = tensorflow.keras.optimizers.Adam(clipnorm=1.0)
        self.DOptimizer = tensorflow.keras.optimizers.Adam(clipnorm=1.0)

        self.GMetric = tensorflow.keras.metrics.BinaryCrossentropy(from_logits=True)
        self.DMetric = tensorflow.keras.metrics.BinaryAccuracy(threshold=0)

    def getDLoss(self, real_output, fake_output):
        real_loss = self.cross_entropy(
            tensorflow.ones_like(real_output), real_output)
        fake_loss = self.cross_entropy(
            tensorflow.zeros_like(fake_output), fake_output)
        return real_loss + fake_loss

    def getGLoss(self, fake_output):
        return self.cross_entropy(tensorflow.ones_like(fake_output), fake_output)

    @tensorflow.function
    def train_step(self, images, GIteration=1, DIteration=1):
        for _ in range(DIteration):
            with tensorflow.GradientTape() as D_tape:
                noise = tensorflow.random.normal(
                    [batchSize, featureVectorLength])
                generated_images = self.G(noise, training=True)
                real_output = self.D(images, training=True)
                fake_output = self.D(generated_images, training=True)
                D_loss = self.getDLoss(real_output, fake_output)

            gradients_of_D = D_tape.gradient(
                D_loss, self.D.trainable_variables)
            self.DOptimizer.apply_gradients(
                zip(gradients_of_D, self.D.trainable_variables))
            
            self.DMetric.update_state(
                tensorflow.ones_like(real_output), real_output)
            self.DMetric.update_state(
                tensorflow.zeros_like(fake_output), fake_output)

        for _ in range(GIteration):
            with tensorflow.GradientTape() as G_tape:
                noise = tensorflow.random.normal(
                    [batchSize, featureVectorLength])
                generated_images = self.G(noise, training=True)
                fake_output = self.D(generated_images, training=True)
                G_loss = self.getGLoss(fake_output)

            gradients_of_G = G_tape.gradient(
                G_loss, self.G.trainable_variables)
            self.GOptimizer.apply_gradients(
                zip(gradients_of_G, self.G.trainable_variables))
            
            self.GMetric.update_state(
                tensorflow.ones_like(fake_output), fake_output)

    def train(self, epochs=1):
        for epoch in range(epochs):
            self.GMetric.reset_state()
            self.DMetric.reset_state()
            self.trainDataReady = False
            self.getData()
            if self.trainDataReady:
                start = time.time()
                for image_batch in self.trainData:
                    self.train_step(image_batch)
                print('Time for epoch {} is {} sec'.format(
                    epoch + 1, time.time()-start))
                print("G Loss: " + str(self.GMetric.result().numpy()))
                print("D Accuracy: " + str(self.DMetric.result().numpy()))
                self.G.save(self.GPath)
                self.D.save(self.DPath)
                self.generateImg()

    def getData(self):
        data_manager.retrieveData()
        self.trainData = data_manager.getDataset()[0]
        if len(self.trainData) > 0:
            self.trainData = tensorflow.data.Dataset.from_tensor_slices(
                self.trainData).shuffle(batchSize).batch(batchSize)
            self.trainDataReady = True
        else:
            print("no data")

    def generateImg(self, num=1):
        img = self.G(tensorflow.random.normal(
            [num, featureVectorLength]))
        img = img * 255
        for index in range(num):
            cv2.imwrite(STATICFILES_DIRS[2] + "/generated" +
                        str(index)+".jpg", numpy.array(img[index, :, :, :]))