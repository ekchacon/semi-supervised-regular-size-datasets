import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from mypackages.readDataset import Normalized as nN

#Calling MNIST dataset
X_train,X_test,y_train,y_test = nN.mnist()

#Taking 10k examples as few labeled examples, which will be gradually decreased in number.
X_train = X_train[50000:] 
y_train = y_train[50000:] 


#Setting the shape of each examples to [28,28].
#rows = 28
#features = 28
#X_train = X_train.reshape((len(y_train),rows,features))
#X_test = X_test.reshape((len(y_test),rows,features))

#The main loop for decreasing from 10k up to 200 labeled examples examples.
for j in range(17,20):
  if j <= 10:
    AmountTrueLab = j*100
  else:
    AmountTrueLab = (j-9)*1000
  
  print(AmountTrueLab)

  datatrain = AmountTrueLab #amount of few labeled examples.
  trainBATCH_SIZE = 32
  testBATCH_SIZE = 512
  BUFFER_SIZE = 10000
  LAYER = 3
  UNITS = 512
  
  #Preparing data to train
  dataXtrain = X_train[:datatrain]
  dataytrain = y_train[:datatrain]
  
  train = tf.data.Dataset.from_tensor_slices((dataXtrain, dataytrain)) # [:datatrain]
  train = train.batch(trainBATCH_SIZE).repeat()
  
  test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
  test = test.batch(testBATCH_SIZE).repeat()
  
  #for each amount of few labeled examples the experiment is repeated 10 times.
  tenRepet = np.empty((1,0), float)
  #loop for ten repetitions 
  for i in range(0,10,1):
    def get_lr_metric(optimizer):
          def lr(y_true, y_pred):
              return optimizer.learning_rate
          return lr
    
    #RNN-LSTM setup
    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=X_train.shape[-2:],dropout=0.0),
        tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
        tf.keras.layers.LSTM(units=UNITS),
        tf.keras.layers.Dense(UNITS, activation='relu'),
        tf.keras.layers.Dense(10,activation='softmax')])
    
    opt = tf.keras.optimizers.Adam(1e-3)
    lr_metric = get_lr_metric(opt)
    model.compile(optimizer=opt,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                  metrics=['acc', lr_metric])
    
    cbks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: (1e-3)/((epoch+1)**(1/2))), 
              tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1),
              ]
    #model.summary()
    
    #training
    stepsPerEpoch = math.floor(dataXtrain.shape[0]/32)
    model.fit(train,epochs=1000, 
                          verbose=0,steps_per_epoch=stepsPerEpoch,  
                          validation_data=test,
                          validation_steps=19, 
                          callbacks=cbks
              )
    
    #getting training results
    results = model.evaluate(x=test, verbose=0, steps=19) 
    if i == 0:
      tenRepet = np.concatenate((tenRepet,dataXtrain.shape[0]), axis=None)
    
    tenRepet = np.concatenate((tenRepet,round(results[1]*100,2)), axis=None)
  
  #Writing results on csv
  import csv
  csv.register_dialect("hashes", delimiter=",")
  f = open('/home/est1/Edgar/Semestre 3/model implementation/realExperiments/supervised/supervised.csv','a')
    
  with f:
      #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
      writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
      writer.writerow(tenRepet)
