import random

import sqlite3
import numpy
import cv2

from myP.settings import STATICFILES_DIRS
from learningModel.setting import batchSize, imageSize

class datasetManager():
    def __init__(self):
        self.trainData = []
        self.trainLabel = []

    def preprocessData(self, path):
        img = cv2.imread(STATICFILES_DIRS[1] +
                         "/" + path)
        if not img is None:
            if img.shape[0] > img.shape[1]:
                border = (img.shape[0] - img.shape[1]) // 2
                img = cv2.copyMakeBorder(
                    img, 0, 0, border, border, random.choice([cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_DEFAULT]))
            elif img.shape[0] < img.shape[1]:
                border = (img.shape[1] - img.shape[0]) // 2
                img = cv2.copyMakeBorder(
                    img, border, border, 0, 0, random.choice([cv2.BORDER_REPLICATE, cv2.BORDER_REFLECT, cv2.BORDER_WRAP, cv2.BORDER_DEFAULT]))
            
            img = cv2.resize(img, (imageSize, imageSize),
                                interpolation=cv2.INTER_AREA)
            return img / 255

    def retrieveData(self):
        connection = sqlite3.connect("db.sqlite3")
        self.trainData = []
        self.trainLabel = []
        numRetry = 0
        while numRetry < 5 and len(self.trainData) < batchSize:
            cursor = connection.execute(
                'select path, label from images_isp order by RANDOM() limit ' + str(batchSize))
            trainDataLable = numpy.array(cursor.fetchall()).T
            for index in range(len(trainDataLable[0])):
                img = self.preprocessData(trainDataLable[0][index])
                if not img is None:
                    self.trainData.append(img)
                    self.trainLabel.append(int(trainDataLable[1][index]))
            numRetry += 1
        connection.close()

    def getDataset(self):
        return self.trainData, self.trainLabel


data_manager = datasetManager()
