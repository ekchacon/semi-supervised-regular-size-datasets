import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import os
from numpy import genfromtxt
import csv

from mypackages.readDataset import Normalized as nN
from mypackages.learningSort import methods as met

#directory for output and input files.
mainDirect = '/home/est1/Edgar/Semestre 3/model implementation/realExperiments/self-training layer-wise/fashion/log/'
pretrainfile = '/home/est1/Edgar/Semestre 3/model implementation/realExperiments/semi-supervised layer-wise/fashion/logPretraining/pretrainingLayerWiseFASHION'

#Calling FASHION dataset
X_train,X_test,y_train,y_test = nN.fashion()

#unlabaled data for pre-training
PX_train = X_train[:50000]

met.preTraining(PX_train,
                X_test,
                mainDirect = mainDirect_,
                namefile = pretrainfile_,
                Epochs = 1000,
                trainBATCH_SIZE = 32,
                testBATCH_SIZE = 512)

#experiments with different amount of few labaled examples.
setTrueLabels = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])

#taking unlabeled data
unlabeledXtrain = X_train[:50000]

#taking labeled data
labeledX = X_train[50000:]
labeledy = y_train[50000:]

#batch size setting
trainBATCH_SIZE = 32
testBATCH_SIZE = 512
unlabBATCH_SIZE = 512
Epochs = 1000

met.selfTrainingLayerWise(mainDirect,
                pretrainfile,
                setTrueLabels,
                unlabeledXtrain,
                labeledX,
                labeledy,
                X_test,
                y_test, 
                trainBATCH_SIZE,
                testBATCH_SIZE, 
                unlabBATCH_SIZE,
                Epochs)

