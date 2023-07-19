import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math

from mypackages.readDataset import Normalized as nN
from mypackages.learningSort import methods as met

#directory for output files.
mainDirect_ = '/home/est1/Edgar/Semestre 3/model implementation/realExperiments/semi-supervised layer-wise/fashion/logPretraining/'
namefile_ = 'semi-supervisedFASHION' #file to write data
pretrainfile_ = 'pretrainingLayerWiseFASHION' #file to load weights

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

#Taking 10k examples as few labeled examples, which will be gradually decreased in number.
X_train = X_train[50000:] 
y_train = y_train[50000:] 

#experiments with different amount of few labaled examples.
setTrueLabels_ = np.array([200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000])

met.semiSupervised(X_train, 
                   X_test, 
                   y_train, 
                   y_test,
                   mainDirect=mainDirect_,
                   namefile=namefile_,
                   pretrainfile=pretrainfile_,
                   Epochs = 1000,
                   trainBATCH_SIZE = 32,
                   testBATCH_SIZE = 512,
                   setTrueLabels = setTrueLabels_) 

