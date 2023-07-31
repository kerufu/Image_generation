import tensorflow
import numpy
import cv2

from learningModel.setting import batchSize, imageSize
from learningModel.datasetManager import data_manager

class Classifier(tensorflow.keras.Model):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = [
            tensorflow.keras.layers.Conv2D(16, 2, activation='selu'),
            tensorflow.keras.layers.Flatten(),
            tensorflow.keras.layers.Dense(32, activation='selu'),
            tensorflow.keras.layers.Dense(1)
        ]

    def call(self, input):
        for l in self.model:
            input = l(input)
        return input


class ClassifierWoker():
    modelPath = "learningModel/savedModel/Classifier"

    def __init__(self):
        self.model = Classifier()
        try:
            self.model.load_weights(self.modelPath)
        except:
            pass
        self.model.compile(
            optimizer=tensorflow.keras.optimizers.Adam(clipnorm=1.0),
            loss=tensorflow.keras.losses.BinaryFocalCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )
        self.trainDataReady = False

    def getData(self):
        data_manager.retrieveData()
        self.trainData, self.trainLabel = data_manager.getDataset()
        if len(self.trainLabel) > 0:
            self.trainData = numpy.float16(self.trainData)
            self.trainLabel = numpy.float16(self.trainLabel)
            self.trainDataReady = True
        else:
            print("no data")


    def train(self, epochs=1):
        for _ in range(epochs):
            self.trainDataReady = False
            self.getData()
            if self.trainDataReady:
                self.model.fit(
                    self.trainData,
                    self.trainLabel,
                    batch_size=batchSize,
                    epochs=1,
                )
                self.model.save(self.modelPath)

    def predict(self, path):
        img = data_manager.preprocessData(path)
        logit = self.model.predict(numpy.float16([img]))
        prediction = float(tensorflow.nn.sigmoid(logit)[0])
        print(logit, prediction)
        return prediction