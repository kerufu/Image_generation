import time

import tensorflow
import numpy
import cv2

from myP.settings import STATICFILES_DIRS
from learningModel.setting import batchSize, imageSize, featureVectorLength, CAAEPath
from learningModel.datasetManager import data_manager

class CrossAttention(tensorflow.keras.layers.Layer):
    def __init__(self, key_dim, num_heads, **kwargs):
        super(CrossAttention, self).__init__()
        
        self.key_dim = key_dim
        self.num_heads = num_heads

        self.x_layer = tensorflow.keras.layers.Dense(self.key_dim*self.num_heads)
        self.q_layer = tensorflow.keras.layers.Dense(self.key_dim*self.num_heads)
        self.k_layer = tensorflow.keras.layers.Dense(self.key_dim*self.num_heads)
        self.v_layer = tensorflow.keras.layers.Dense(self.key_dim*self.num_heads)
        self.mha = tensorflow.keras.layers.MultiHeadAttention(key_dim=self.key_dim, num_heads=self.num_heads, **kwargs)
        self.layernorm = tensorflow.keras.layers.LayerNormalization()
        self.add = tensorflow.keras.layers.Add()

    def call(self, x, context=None):

        x = self.x_layer(x)

        if context is None:
            context = x

        q, k, v = self.q_layer(x), self.k_layer(context), self.v_layer(context)

        q = tensorflow.reshape(q, (-1, self.num_heads, self.key_dim))
        k = tensorflow.reshape(k, (-1, self.num_heads, self.key_dim))
        v = tensorflow.reshape(v, (-1, self.num_heads, self.key_dim))

        attn_output = self.mha(query=q, key=k, value=v)
        attn_output = tensorflow.reshape(attn_output, (-1, self.num_heads*self.key_dim))
        x = self.add([x, attn_output])
        x = self.layernorm(x)
        return x
        
class Encoder(tensorflow.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = [
            tensorflow.keras.layers.Conv2D(8, 2, activation='selu', padding='same'),
            tensorflow.keras.layers.Conv2D(16, 2, activation='selu', padding='same'),
            tensorflow.keras.layers.Flatten(),
            CrossAttention(128, 8),
            tensorflow.keras.layers.Dense(featureVectorLength)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class DiscriminatorOnEncoder(tensorflow.keras.Model):
    def __init__(self):
        super(DiscriminatorOnEncoder, self).__init__()
        self.model = [
            tensorflow.keras.layers.Dense(8, activation='selu'),
            tensorflow.keras.layers.Dense(1)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class Decoder(tensorflow.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = [
            CrossAttention(128, 8),
            tensorflow.keras.layers.Dense(imageSize*imageSize, activation='selu'),
            tensorflow.keras.layers.Reshape((imageSize//4, imageSize//4, 16)),
            tensorflow.keras.layers.Conv2DTranspose(8, 2, strides=2, padding='same', activation='selu'),
            tensorflow.keras.layers.Conv2DTranspose(3, 2, strides=2, padding='same')
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class DiscriminatorOnDecoder(tensorflow.keras.Model):
    def __init__(self):
        super(DiscriminatorOnDecoder, self).__init__()
        self.model = [
            tensorflow.keras.layers.Conv2D(8, 2, activation='selu'),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(1)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class CAAEWoker():
    EPath, DOEPath, DPath, DODPath = CAAEPath

    def __init__(self):
        self.E = Encoder()
        self.DOE = DiscriminatorOnEncoder()
        self.D = Decoder()
        self.DOD = DiscriminatorOnDecoder()
        try:
            self.E.load_weights(self.EPath)
            self.DOE.load_weights(self.DOEPath)
            self.D.load_weights(self.DPath)
            self.DOD.load_weights(self.DODPath)
            print("CAAE weight loaded")
        except:
            print("CAAE weight not found")
        self.trainDataReady = False
        self.cross_entropy = tensorflow.keras.losses.BinaryFocalCrossentropy(from_logits=True)
        self.mse = tensorflow.keras.losses.MeanSquaredError()

        self.EOptimizer = tensorflow.keras.optimizers.Adam(clipnorm=1.0)
        self.DOEOptimizer = tensorflow.keras.optimizers.Adam(clipnorm=1.0)
        self.DOptimizer = tensorflow.keras.optimizers.Adam(clipnorm=1.0)
        self.DODOptimizer = tensorflow.keras.optimizers.Adam(clipnorm=1.0)

        self.AutoEncodeMetric = tensorflow.keras.metrics.MeanSquaredError()
        self.DOEMetric = tensorflow.keras.metrics.BinaryAccuracy(threshold=0)
        self.DODMetric = tensorflow.keras.metrics.BinaryAccuracy(threshold=0)

    def getELoss(self, DOE_fake_output, DOD_fake_output, input_image, D_output):
        discriminator_loss = self.cross_entropy(
            tensorflow.ones_like(DOE_fake_output), DOE_fake_output)
        discriminator_loss += self.cross_entropy(
            tensorflow.ones_like(DOD_fake_output), DOD_fake_output)
        image_loss = self.mse(input_image, D_output)
        return discriminator_loss + image_loss

    def getDOELoss(self, DOE_real_output, DOE_fake_output):
        real_loss = self.cross_entropy(
            tensorflow.ones_like(DOE_real_output), DOE_real_output)
        fake_loss = self.cross_entropy(
            tensorflow.zeros_like(DOE_fake_output), DOE_fake_output)
        return real_loss + fake_loss

    def getDLoss(self, DOD_fake_output, input_image, D_output):
        discriminator_loss = self.cross_entropy(
            tensorflow.ones_like(DOD_fake_output), DOD_fake_output)
        image_loss = self.mse(input_image, D_output)
        return discriminator_loss + image_loss

    def getDODLoss(self, DOD_real_output, DOD_fake_output):
        real_loss = self.cross_entropy(
            tensorflow.ones_like(DOD_real_output), DOD_real_output)
        fake_loss = self.cross_entropy(
            tensorflow.zeros_like(DOD_fake_output), DOD_fake_output)
        return real_loss + fake_loss

    @tensorflow.function
    def train_step(self, images, conditions, AEIteration=1, DOEIteration=1, DODIteration=1):
        for _ in range(DOEIteration):
            with tensorflow.GradientTape() as DOE_tape:
                noise = tensorflow.random.uniform(
                    [batchSize, featureVectorLength])
                encoded_feature_vector = self.E(images, training=True)
                DOE_real_output = self.DOE(noise, training=True)
                DOE_fake_output = self.DOE(encoded_feature_vector, training=True)
                DOE_loss = self.getDOELoss(DOE_real_output, DOE_fake_output)

            gradients_of_DOE = DOE_tape.gradient(
                DOE_loss, self.DOE.trainable_variables)
            self.DOEOptimizer.apply_gradients(
                zip(gradients_of_DOE, self.DOE.trainable_variables))
            self.DOEMetric.update_state(tensorflow.ones_like(DOE_real_output), DOE_real_output)
            self.DOEMetric.update_state(tensorflow.zeros_like(DOE_fake_output), DOE_fake_output)
        for _ in range(DODIteration):
            with tensorflow.GradientTape() as DOD_tape:
                encoded_feature_vector = self.E(images, training=True)
                conditional_encoded_feature_vector = tensorflow.concat(
                    [encoded_feature_vector, conditions], 1)
                decoded_images = self.D(
                    conditional_encoded_feature_vector, training=True)
                DOD_real_output = self.DOD(images, training=True)
                DOD_fake_output = self.DOD(decoded_images, training=True)
                DOD_loss = self.getDODLoss(DOD_real_output, DOD_fake_output)

            gradients_of_DOD = DOD_tape.gradient(
                DOD_loss, self.DOD.trainable_variables)
            self.DODOptimizer.apply_gradients(
                zip(gradients_of_DOD, self.DOD.trainable_variables))
            
            self.DODMetric.update_state(
                tensorflow.ones_like(DOD_real_output), DOD_real_output)
            self.DODMetric.update_state(
                tensorflow.zeros_like(DOD_fake_output), DOD_fake_output)

        for _ in range(AEIteration):
            with tensorflow.GradientTape() as E_tape:
                with tensorflow.GradientTape() as D_tape:
                    encoded_feature_vector = self.E(images, training=True)
                    DOE_fake_output = self.DOE(
                        encoded_feature_vector, training=True)
                    conditional_encoded_feature_vector = tensorflow.concat(
                        [encoded_feature_vector, conditions], 1)
                    decoded_images = self.D(
                        conditional_encoded_feature_vector, training=True)
                    E_loss = self.getELoss(
                        DOE_fake_output, DOD_fake_output, images, decoded_images)
                    DOD_fake_output = self.DOD(decoded_images, training=True)
                    D_loss = self.getDLoss(
                        DOD_fake_output, images, decoded_images)

            gradients_of_E = E_tape.gradient(
                E_loss, self.E.trainable_variables)
            self.EOptimizer.apply_gradients(
                zip(gradients_of_E, self.E.trainable_variables))
            gradients_of_D = D_tape.gradient(
                D_loss, self.D.trainable_variables)
            self.DOptimizer.apply_gradients(
                zip(gradients_of_D, self.D.trainable_variables))
            
            self.AutoEncodeMetric.update_state(images, decoded_images)

    def train(self, epochs=1):
        for epoch in range(epochs):
            self.AutoEncodeMetric.reset_state()
            self.DOEMetric.reset_state()
            self.DODMetric.reset_state()

            self.trainDataReady = False
            self.getData()
            if self.trainDataReady:
                start = time.time()
                for batch in self.trainData:
                    images, conditions = batch
                    conditions = tensorflow.expand_dims(tensorflow.cast(
                        conditions, dtype=tensorflow.float32), axis=1)
                    self.train_step(images, conditions)
                print('Time for epoch {} is {} sec'.format(
                    epoch + 1, time.time()-start))
                print("AE Loss: " + str(self.AutoEncodeMetric.result().numpy()))
                print("DOE Accuracy: " + str(self.DOEMetric.result().numpy()))
                print("DOD Accuracy: " + str(self.DODMetric.result().numpy()))
                self.E.save(self.EPath)
                self.DOE.save(self.DOEPath)
                self.D.save(self.DPath)
                self.DOD.save(self.DODPath)
                self.generateImg()

    def getData(self):
        data_manager.retrieveData()
        self.trainData, self.trainLabel = data_manager.getDataset()
        if len(self.trainData) > 0:
            self.trainData = tensorflow.data.Dataset.from_tensor_slices(
                (self.trainData, self.trainLabel)).shuffle(batchSize).batch(batchSize)
            self.trainDataReady = True
        else:
            print("no data")

    def generateImg(self, num=1, condition=1):
        feature = tensorflow.random.uniform([num, featureVectorLength])
        feature = tensorflow.concat([feature, [[condition]]], 1)
        img = self.D(feature)
        img = img * 255
        for index in range(num):
            cv2.imwrite(STATICFILES_DIRS[2] + "/generated" +
                        str(index)+".jpg", numpy.array(img[index, :, :, :]))