from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
from django.urls import reverse
import os
import threading

from images.models import pI, isP
from learningModel.Classifier import ClassifierWoker
from learningModel.StandardGAN import StandardGANWoker
from learningModel.CAAE import CAAEWoker

from myP.settings import STATICFILES_DIRS

classifierWoker = ClassifierWoker()
standardGANWoker = StandardGANWoker()
caaeWoker = CAAEWoker()
isTraining = False
def index(request):
    template = loader.get_template('index.html')
    pITableLength = pI.objects.all().count()
    isPTableLength = isP.objects.all().count()
    context = {
        'pITableLength': pITableLength,
        'isPTableLength': isPTableLength
    }
    return HttpResponse(template.render(context, request))


def updateDB(request):
    pI.objects.all().delete()
    def getListOfFiles(dirName):
        # create a list of file and sub directories
        # names in the given directory
        listOfFile = os.listdir(dirName)
        allFiles = list()
        # Iterate over all the entries
        for entry in listOfFile:
            # Create full path
            if entry[0] != "." and entry[0] != "_": 
                fullPath = os.path.join(dirName, entry)
                # If entry is a directory then get the list of files in this directory
                if os.path.isdir(fullPath):
                    allFiles = allFiles + getListOfFiles(fullPath)
                elif fullPath.endswith('.png') or fullPath.endswith('.jpg') or fullPath.endswith('.jpeg'):
                    allFiles.append(os.path.relpath(fullPath, STATICFILES_DIRS[1]))

        return allFiles

    for p in getListOfFiles(STATICFILES_DIRS[1]):
        pathObj = pI(path=p)
        pathObj.save()

    return HttpResponseRedirect('/images/')


def show(request, withPrediciton=False):
    imageToShow = pI.objects.values_list('path').order_by('?').first()[0]
    isPPrediction = -1
    if withPrediciton == 1 and (not isTraining):
        count = 0
        while isPPrediction < 0.5 and count < 10:
            imageToShow = pI.objects.values_list(
                'path').order_by('?').first()[0]
            isPPrediction = classifierWoker.predict(imageToShow)
            count += 1
    template = loader.get_template('show.html')
    context = {
        'imageToShow': imageToShow,
        'isPPrediction': isPPrediction,
        'withPrediciton': withPrediciton
    }
    return HttpResponse(template.render(context, request))


def labelImage(request, withPrediciton=False):
    imagePath, imageLabel = request.POST.get('path'), request.POST.get('isP')
    isPObj = isP(path=imagePath, label=imageLabel)
    isPObj.save()
    return HttpResponseRedirect('/images/show/'+str(withPrediciton))

def learnImage(request):
    
    def learnFunction():
        global isTraining
        isTraining = True
        # classifierWoker.train()
        standardGANWoker.train(100)
        # caaeWoker.train()
        isTraining = False

    if not isTraining:
        learnThread = threading.Thread(target=learnFunction)
        learnThread.start()
    return HttpResponseRedirect(reverse('index'))


def generateImage(request, isCAAE=False):
    if not isTraining:
        if isCAAE:
            caaeWoker.generateImg()
        else:
            standardGANWoker.generateImg()
    template = loader.get_template('generate.html')
    context = {
        'imageToShow': "generated0.jpg"
    }
    return HttpResponse(template.render(context, request))


def clearDataset(request):
    isP.objects.all().delete()
    return HttpResponseRedirect(reverse('index'))