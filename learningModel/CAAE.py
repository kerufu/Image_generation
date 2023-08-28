import time

import tensorflow as tf
import numpy
import cv2

from myP.settings import STATICFILES_DIRS
from learningModel.setting import batchSize, imageSize, featureVectorLength, CAAEPath
from learningModel.datasetManager import data_manager

class DenseCrossAttention(tf.keras.layers.Layer):
    def __init__(self, key_dim, num_heads, units):
        super(DenseCrossAttention, self).__init__()
        
        self.key_dim = key_dim
        self.num_heads = num_heads

        self.x_layer = tf.keras.layers.Dense(self.key_dim*self.num_heads)
        self.q_layer = tf.keras.layers.Dense(self.key_dim*self.num_heads)
        self.k_layer = tf.keras.layers.Dense(self.key_dim*self.num_heads)
        self.v_layer = tf.keras.layers.Dense(self.key_dim*self.num_heads)
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=self.key_dim, num_heads=self.num_heads)
        self.attn_layer = tf.keras.layers.Dense(self.key_dim*self.num_heads)
        self.add = tf.keras.layers.Add()
        self.output_layer = tf.keras.layers.Dense(units)
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x, context=None):

        x = self.x_layer(x)

        if context is None:
            context = x

        q, k, v = self.q_layer(x), self.k_layer(context), self.v_layer(context)
        q = tf.reshape(q, (-1, self.num_heads, self.key_dim))
        k = tf.reshape(k, (-1, self.num_heads, self.key_dim))
        v = tf.reshape(v, (-1, self.num_heads, self.key_dim))

        attn_output = self.mha(query=q, key=k, value=v)
        attn_output = tf.reshape(attn_output, (-1, self.num_heads*self.key_dim))
        attn_output = self.attn_layer(attn_output)

        x = self.add([x, attn_output])
        x = self.output_layer(x)
        x = self.layernorm(x)
        
        return x
    
class Conv2DCrossAttention(tf.keras.layers.Layer):
    def __init__(self, key_dim, num_heads, filters, kernel_size, context_size=16, context_filters=16):
        super(Conv2DCrossAttention, self).__init__()
        
        self.key_dim = key_dim
        self.num_heads = num_heads

        self.reshape_layers = [
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(context_size*context_size*context_filters),
            tf.keras.layers.Reshape([-1, context_size, context_size, context_filters])
        ]
        self.q_layer = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.k_layer = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.v_layer = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.mha = tf.keras.layers.MultiHeadAttention(key_dim=key_dim, num_heads=num_heads, attention_axes=(1, 2))
        self.attn_layer = tf.keras.layers.Conv2D(filters, kernel_size, padding='same')
        self.add = tf.keras.layers.Add()
        self.layernorm = tf.keras.layers.LayerNormalization()

    def call(self, x, context=None):

        if context is None:
            context = x

        if len(context.shape) < 4:
            for rl in self.reshape_layers:
                context = rl(context)

        q, k, v = self.q_layer(x), self.k_layer(context), self.v_layer(context)

        attn_output = self.mha(query=q, key=k, value=v)
        attn_output = self.attn_layer(attn_output)

        x = self.add([x, attn_output])
        x = self.layernorm(x)
        
        return x
        
class Encoder(tf.keras.Model):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = [
            tf.keras.layers.Conv2D(32, 5, activation='selu', padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Conv2D(64, 3, activation='selu', padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Conv2D(128, 2, activation='selu', padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Conv2D(256, 2, activation='selu', padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Flatten(),
            DenseCrossAttention(128, 8, 2048),
            tf.keras.layers.Dense(featureVectorLength)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class DiscriminatorOnEncoder(tf.keras.Model):
    def __init__(self):
        super(DiscriminatorOnEncoder, self).__init__()
        self.model = [
            tf.keras.layers.Dense(256, activation='selu', kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Dense(64, activation='selu', kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Dense(1)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class Decoder(tf.keras.Model):
    def __init__(self):
        super(Decoder, self).__init__()
        self.model = [
            DenseCrossAttention(128, 8, 2048),
            tf.keras.layers.Dense(imageSize*imageSize*4, activation='selu', kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Reshape((imageSize//4, imageSize//4, 256)),
            tf.keras.layers.Conv2DTranspose(128, 2, strides=2, padding='same', activation='selu', kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Conv2DTranspose(64, 2, strides=2, padding='same', activation='selu', kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Conv2DTranspose(32, 3, strides=2, padding='same', activation='selu', kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Conv2DTranspose(3, 5, strides=2, padding='same', kernel_initializer=tf.keras.initializers.LecunNormal())
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class DiscriminatorOnDecoder(tf.keras.Model):
    def __init__(self):
        super(DiscriminatorOnDecoder, self).__init__()
        self.model = [
            tf.keras.layers.Conv2D(256, 3, activation='selu', padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Conv2D(128, 2, activation='selu', padding='same', dilation_rate=2, kernel_initializer=tf.keras.initializers.LecunNormal()),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(1)
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
        self.cross_entropy = tf.keras.losses.BinaryFocalCrossentropy(from_logits=True)
        self.mse = tf.keras.losses.MeanSquaredError()

        self.EOptimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
        self.DOEOptimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
        self.DOptimizer = tf.keras.optimizers.Adam(clipnorm=1.0)
        self.DODOptimizer = tf.keras.optimizers.Adam(clipnorm=1.0)

        self.AutoEncodeMetric = tf.keras.metrics.MeanSquaredError()
        self.DOEMetric = tf.keras.metrics.BinaryAccuracy(threshold=0)
        self.DODMetric = tf.keras.metrics.BinaryAccuracy(threshold=0)

    def getELoss(self, DOE_fake_output, DOD_fake_output, input_image, D_output):
        discriminator_loss = self.cross_entropy(
            tf.ones_like(DOE_fake_output), DOE_fake_output)
        discriminator_loss += self.cross_entropy(
            tf.ones_like(DOD_fake_output), DOD_fake_output)
        image_loss = self.mse(input_image, D_output)
        return discriminator_loss + image_loss

    def getDOELoss(self, DOE_real_output, DOE_fake_output):
        real_loss = self.cross_entropy(
            tf.ones_like(DOE_real_output), DOE_real_output)
        fake_loss = self.cross_entropy(
            tf.zeros_like(DOE_fake_output), DOE_fake_output)
        return real_loss + fake_loss

    def getDLoss(self, DOD_fake_output, input_image, D_output):
        discriminator_loss = self.cross_entropy(
            tf.ones_like(DOD_fake_output), DOD_fake_output)
        image_loss = self.mse(input_image, D_output)
        return discriminator_loss + image_loss

    def getDODLoss(self, DOD_real_output, DOD_fake_output):
        real_loss = self.cross_entropy(
            tf.ones_like(DOD_real_output), DOD_real_output)
        fake_loss = self.cross_entropy(
            tf.zeros_like(DOD_fake_output), DOD_fake_output)
        return real_loss + fake_loss

    @tf.function
    def train_step(self, images, conditions, AEIteration=1, DOEIteration=1, DODIteration=1):
        for _ in range(DOEIteration):
            with tf.GradientTape() as DOE_tape:
                noise = tf.random.uniform(
                    [batchSize, featureVectorLength], minval=-1, maxval=1)
                encoded_feature_vector = self.E(images, training=True)
                DOE_real_output = self.DOE(noise, training=True)
                DOE_fake_output = self.DOE(encoded_feature_vector, training=True)
                DOE_loss = self.getDOELoss(DOE_real_output, DOE_fake_output)

            gradients_of_DOE = DOE_tape.gradient(
                DOE_loss, self.DOE.trainable_variables)
            self.DOEOptimizer.apply_gradients(
                zip(gradients_of_DOE, self.DOE.trainable_variables))
            self.DOEMetric.update_state(tf.ones_like(DOE_real_output), DOE_real_output)
            self.DOEMetric.update_state(tf.zeros_like(DOE_fake_output), DOE_fake_output)
        for _ in range(DODIteration):
            with tf.GradientTape() as DOD_tape:
                encoded_feature_vector = self.E(images, training=True)
                conditional_encoded_feature_vector = tf.concat(
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
                tf.ones_like(DOD_real_output), DOD_real_output)
            self.DODMetric.update_state(
                tf.zeros_like(DOD_fake_output), DOD_fake_output)

        for _ in range(AEIteration):
            with tf.GradientTape() as E_tape:
                with tf.GradientTape() as D_tape:
                    encoded_feature_vector = self.E(images, training=True)
                    DOE_fake_output = self.DOE(
                        encoded_feature_vector, training=True)
                    conditional_encoded_feature_vector = tf.concat(
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
                    conditions = tf.expand_dims(tf.cast(
                        conditions, dtype=tf.float32), axis=1)
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
            self.trainData = tf.data.Dataset.from_tensor_slices(
                (self.trainData, self.trainLabel)).shuffle(batchSize).batch(batchSize)
            self.trainDataReady = True
        else:
            print("no data")

    def generateImg(self, num=1, condition=1):
        feature = tf.random.uniform([num, featureVectorLength], minval=-1, maxval=1)
        feature = tf.concat([feature, [[condition]]], 1)
        img = self.D(feature)
        img = img * 255
        for index in range(num):
            cv2.imwrite(STATICFILES_DIRS[2] + "/generated" +
                        str(index)+".jpg", numpy.array(img[index, :, :, :]))