import tensorflow as tf
import numpy as np
import math
from mypackages.learningSort import methods as met
import os
import csv
from numpy import genfromtxt

def semiSupervised(X_train, 
                   X_test, 
                   y_train, 
                   y_test,
                   mainDirect='',
                   namefile='',
                   pretrainfile='',
                   Epochs = 1000,
                   trainBATCH_SIZE = 32,
                   testBATCH_SIZE = 512,
                   #numberTrueLabeles = 2,
                   setTrueLabels = np.array([],dtype=int)):

  #Seq Length
  features = X_train.shape[2]
  #X_train = X_train.reshape((len(y_train),rows,features))
  #X_test = X_test.reshape((len(y_test),rows,features))
  
  for AmountTrueLab in setTrueLabels:
  
    print(AmountTrueLab)
    tf.keras.backend.clear_session()
  
    #Epochs = 1000
    datatrain = AmountTrueLab
    #trainBATCH_SIZE =32
    #testBATCH_SIZE = 512
    BUFFER_SIZE = 10000
    LAYER = 3
    UNITS = 512
    
    ######Data to train
    dataXtrain = X_train[:datatrain]
    dataytrain = y_train[:datatrain]
    
    train = tf.data.Dataset.from_tensor_slices((dataXtrain, dataytrain))
    train = train.batch(trainBATCH_SIZE).repeat()
    
    test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
    test = test.batch(testBATCH_SIZE).repeat()
    
    tenRepet = np.empty((1,0), float)
    #loop for ten repetitions 
    for i in range(0,10,1):
    
      def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer.learning_rate
            return lr
      
      #RNN-LSTM setup for training
      model = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=X_train.shape[-2:],dropout=0.0),
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
          tf.keras.layers.LSTM(units=UNITS),
          tf.keras.layers.Dense(features),
          #tf.keras.layers.Dense(28, activation='relu'),
          tf.keras.layers.Dense(10,activation='softmax')])#25 for slmnist    
      
      opt = tf.keras.optimizers.Adam(1e-3)
      lr_metric = get_lr_metric(opt)
      model.compile(optimizer=opt,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                    metrics=['acc', lr_metric])
      #model.summary()
      
      cbks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: (1e-3)/((epoch+1)**(1/2))), #(1e-3)/(epoch+1) #0.001
                #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time())),
                tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1),
                tf.keras.callbacks.CSVLogger(mainDirect+'log_'+namefile+'.csv', append=True, separator=',')
                ]
      #model.summary()
      
      #model.layers[0].get_weights()
      
      #RNN-LSTM setup to load pre-trained weights
      pretrainedModel = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=X_train.shape[-2:],dropout=0.0),
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
          tf.keras.layers.LSTM(units=UNITS,return_sequences=True),
          tf.keras.layers.Dense(features)
      ])
      
      #pretrainedModel.summary()
      
      #loading pre-trained weights
      pretrainedModel.load_weights(mainDirect+pretrainfile+'.h5')
      
      pretrainedModel.layers[0].get_weights()
      
      model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
      model.layers[1].set_weights(pretrainedModel.layers[1].get_weights())
      model.layers[2].set_weights(pretrainedModel.layers[2].get_weights())
      model.layers[3].set_weights(pretrainedModel.layers[3].get_weights())
      
      #training
      stepsPerEpoch = math.floor(X_train[:datatrain].shape[0]/trainBATCH_SIZE)
      stepsTest = math.floor(X_test.shape[0]/testBATCH_SIZE)
      model.fit(train,epochs=Epochs, 
                            verbose=0,steps_per_epoch=stepsPerEpoch, 
                            validation_data=test,
                            validation_steps=stepsTest, 
                            callbacks=cbks
                )
      
      #getting training results
      results = model.evaluate(x=test,  verbose=0, steps=stepsTest)
      
      if i == 0:
        tenRepet = np.concatenate((tenRepet,dataXtrain.shape[0]), axis=None)
          
      tenRepet = np.concatenate((tenRepet,round(results[1]*100,2)), axis=None)
    
    #Writing results on csv
    import csv
    csv.register_dialect("hashes", delimiter=",")
    f = open(mainDirect+namefile+'AccuTenTimes.csv','a')
        
    with f:
        #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
        writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
        writer.writerow(tenRepet)

def preTraining(X_train,
                X_test,
                mainDirect,
                namefile,
                Epochs = 1000,
                trainBATCH_SIZE = 32,
                testBATCH_SIZE = 512):

  BUFFER_SIZE = 10000
  UNITS = 512
  
  #Seq Length
  features = X_train.shape[2]
  
  ######Data to train
  train = tf.data.Dataset.from_tensor_slices((X_train,X_train)) 
  train = train.cache().shuffle(BUFFER_SIZE).batch(trainBATCH_SIZE).repeat()
  
  test = tf.data.Dataset.from_tensor_slices((X_test, X_test))
  test = test.batch(testBATCH_SIZE).repeat()
  
  #Layer-wise pre-training-------
  for L in range(1,4,1): #4
    print(L)
  
    def get_lr_metric(optimizer):
          def lr(y_true, y_pred):
              return optimizer.learning_rate
          return lr
  
    if L == 1:
      model = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=X_train.shape[-2:],dropout=0.0),
         #tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
         #tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
         tf.keras.layers.Dense(features)
      ])
    
    if L == 2:
      model = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=X_train.shape[-2:],dropout=0.0),
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
         #tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
         tf.keras.layers.Dense(features)
      ])
    
      pretrainedModel = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=X_train.shape[-2:],dropout=0.0),
          #tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
         #tf.keras.layers.LSTM(units=UNITS,return_sequences=True),
         tf.keras.layers.Dense(features) 
      ])
    
    if L == 3:
      model = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=X_train.shape[-2:],dropout=0.0),
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
         tf.keras.layers.Dense(features) 
      ])
    
      pretrainedModel = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=X_train.shape[-2:],dropout=0.0),
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
         #tf.keras.layers.LSTM(units=UNITS,return_sequences=True),
         tf.keras.layers.Dense(features) 
      ])
    
    #Compile
    opt = tf.keras.optimizers.Adam(1e-3)
    lr_metric = get_lr_metric(opt)
    model.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy',lr_metric]) 
    #model.summary()
    
    cbks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: (1e-3)/((epoch+1)**(1/2))), #(1e-3)/(epoch+1) #0.001
              #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time())),
              tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1),
              tf.keras.callbacks.CSVLogger(mainDirect+'log_'+namefile+'.csv', append=True, separator=',')
              ]
    
    ##if 1L 
      ##No load
    
    ##if 2L then Load
    if L == 2:
      pretrainedModel.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy',lr_metric])
      pretrainedModel.load_weights(mainDirect+str(namefile)+'.h5')
      #L2ToPretrain[1]( L1Pretrained [0] and [1]Dense)
      model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
      model.layers[2].set_weights(pretrainedModel.layers[1].get_weights())

      #print("***")
      model.layers[0].trainable = False
      #model.summary()
    
    ##if 3L then Load
    if L == 3:
      pretrainedModel.compile(optimizer=opt, loss=tf.keras.losses.MeanSquaredError(), metrics=['accuracy',lr_metric])
      pretrainedModel.load_weights(mainDirect+str(namefile)+'.h5')
      #L3ToPretrain[2]( L1L2Pretrained [0], [1] and [2]Dense)
      model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
      model.layers[1].set_weights(pretrainedModel.layers[1].get_weights())
      model.layers[3].set_weights(pretrainedModel.layers[2].get_weights())

      #print("***")
      model.layers[0].trainable = False
      model.layers[1].trainable = False
      #model.summary()
    
    #training
    stepsPerEpoch = math.floor(X_train.shape[0]/trainBATCH_SIZE)
    stepsTest = math.floor(X_test.shape[0]/testBATCH_SIZE)
    model.fit(train,epochs=Epochs, 
                          verbose=0,steps_per_epoch=stepsPerEpoch, 
                          validation_data=test,
                          validation_steps=stepsTest, 
                          callbacks=cbks
              )
    
    model.save_weights(mainDirect+str(namefile)+'.h5')
    #Layer-wise pre-training-------
    
    #getting training results
    datamodel = model.evaluate( x=test, verbose=1, steps=stepsTest) #

    #Writing results on csv
    import csv
    csv.register_dialect("hashes", delimiter=",")
    f = open(mainDirect+'evaluation_'+str(namefile)+'.csv','a')
    
    with f:
        #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
        writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
        writer.writerow(datamodel)

def transferUnlabToAug(unlabX_train,
                        augX_train,
                        augy_train,
                        accuPerExample,
                        labelPerExample):
  
  #number of examples to get per class.
  numberExamplesPerClass = int(10 ** ((math.log10( augX_train.shape[0] ) // 1) - 1)) #Define how to get examples in according to the amount of examples (ten, hundred, thousands, ...).
  
  #Choosing the best predictions.  
  exampleToMove = np.empty((1,0), int)
  for cat in range(0,10,1):
    #print(cat)
  
    label = np.array( np.where(labelPerExample == cat) ).flatten() #give a cat indices
    #label.shape
    
    #try:
    ind = np.argpartition(accuPerExample[label],-numberExamplesPerClass)[-numberExamplesPerClass:]# N largest numbers
    #except ValueError:  #raised if `y` is empty.
    #    pass
    #    break
        
    #try:
    minAccu = np.amin(accuPerExample[label][ind]) # min accu
    #except ValueError:  #raised if `y` is empty.
    #    pass
    #    break
    
    accu = np.array( np.where(accuPerExample >= minAccu) ).flatten()
    #accu.shape
  
    accuLabelOK = np.intersect1d(accu,label)[:numberExamplesPerClass] #10 examples or 100, it depends on ...
  
    #labelPerExample[accuLabelOK]
    #accuPerExample[accuLabelOK]
  
    exampleToMove = np.concatenate((exampleToMove,accuLabelOK), axis=None)
  #labelPerExample[exampleToMove].shape
  
  #How to move examples from unlabeled into labeled dataset
  #try:
  augX_train = np.concatenate((augX_train,unlabX_train[exampleToMove]), axis=0)
  #except ValueError:  #raised if `y` is empty.
  #    pass
  #    break 
  
  #augX_train[201]
  #plt.imshow(augX_train[291])
  
    #Concat y_train
  augy_train = np.concatenate((augy_train,labelPerExample[exampleToMove]), axis=None)
  #augy_train[299]
  
  #augX_train.shape
  #augy_train.shape
  
  #Delete transfered examples from original dataset
  unlabX_train = np.delete(unlabX_train, exampleToMove, axis = 0)
  #unlaby_train = np.delete(unlaby_train, exampleToMove) #I never use unlaby_train, here only to delete.
  
  return unlabX_train, augX_train, augy_train

def selfTraining(mainDirect,
                setTrueLabels,
                unlabeledXtrain,
                labeledX,
                labeledy,
                X_test,
                y_test, 
                trainBATCH_SIZE,
                testBATCH_SIZE, 
                unlabBATCH_SIZE,
                Epochs):

  for AmountTrueLab in setTrueLabels:
    print(AmountTrueLab)
    #AmountTrueLab = 200
  
  #creating folders to save results and weigths
    try:
        os.makedirs(mainDirect+str(AmountTrueLab))
    except FileExistsError:
          # directory already exists
        pass
    
    for tenTimes in range(0,10,1):
      tf.keras.backend.clear_session()
      try:
          os.makedirs(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes))
      except FileExistsError:
            # directory already exists
          pass
      
      #X_train.shape
      #X_test.shape
      
      #unlabeled data
      unlabX_train = unlabeledXtrain
      #unlaby_train = y_train[:50000]
      
      #labeled data
      labX_train = labeledX
      laby_train = labeledy
      
      #amount of true labels
      datatrain = AmountTrueLab #200,600, AmountTrueLab
      #trainBATCH_SIZE = 32
      #testBATCH_SIZE = 512
      #unlabBATCH_SIZE = 512
      
      BUFFER_SIZE = 10000
      LAYER = 3
      UNITS = 512
      
      #It is not within loop
      
      def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer.learning_rate
            return lr
      
      #RNN-LSTM setup
      model = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=labX_train.shape[-2:],dropout=0.0),
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
          tf.keras.layers.LSTM(units=UNITS),
          tf.keras.layers.Dense(UNITS, activation='relu'),
          tf.keras.layers.Dense(10,activation='softmax')])#25 for slmnist
      
      opt = tf.keras.optimizers.Adam(1e-3)
      lr_metric = get_lr_metric(opt)
      model.compile(optimizer=opt,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                    metrics=['acc', lr_metric])
      #model.summary()
      
      #Augmented training examples
      augX_train = labX_train[:datatrain]
      augy_train = laby_train[:datatrain]
      
      cbks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: (1e-3)/((epoch+1)**(1/2))), #(1e-3)/(epoch+1) #0.001
                #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time())),
                tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1),
                tf.keras.callbacks.CSVLogger(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/'+str(augX_train.shape[0])+'.csv', append=True, separator=',')
                ]#(write_graph=True)] # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
      
      #Loop (if i >= 8 then numberExamplesPerClass is 1000)
      while (unlabX_train.shape[0] != 0): #LOOP to add PSEUDO-LABELS INTO TRUE LABELS unlabX_train.shape[0]
        #print(unlabX_train.shape)
        #augX_train.shape
        #augy_train.shape
        
        
        unlabeled = tf.data.Dataset.from_tensor_slices((unlabX_train)) # [:datatrain]
        unlabeled = unlabeled.batch(unlabBATCH_SIZE)
        
        train = tf.data.Dataset.from_tensor_slices((augX_train, augy_train)) # [:datatrain]
        train = train.batch(trainBATCH_SIZE).repeat()
        
        test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test = test.batch(testBATCH_SIZE).repeat()
        
        #training
        stepsPerEpoch = math.floor(augX_train.shape[0]/trainBATCH_SIZE)
        stepsTest = math.floor(X_test.shape[0]/testBATCH_SIZE)
        model.fit(train,epochs=Epochs, 
                              verbose=0,steps_per_epoch=stepsPerEpoch,  #58 for 60k (training data) entre 1024 (batch zise) = 58.59
                              validation_data=test,
                              validation_steps=stepsTest, #9 for 10k (test data) entre 1024 (batchsize)
                              callbacks=cbks
                  )
        model.save_weights(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/'+str(augX_train.shape[0])+'.h5')
        
        #model.load_weights('/content/drive/My Drive/PhD/Semestre 3/model implementation/LSTM semi-supervised/self-training/weights/'+str(augX_train.shape[0])+'.h5')
        
        #pred = model.predict(unlabeled,steps=19).shape # batch_size=unlabBATCH_SIZE,

        #getting the results
        results = model.evaluate(x=test, verbose=1, steps=stepsTest) #batch_size=testBATCH_SIZE,
        amodel = [augX_train.shape[0], round(results[1]*100,2)]
        
        ## writing results on csv
        csv.register_dialect("hashes", delimiter=",")
        f = open(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/selftrain.csv','a')
        
        with f:
            #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
            writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
            writer.writerow(amodel)
        ## CSV writing
      
        #Getting Accu and labels per example from predictions.
        #count = 0
        accuPerExample = np.empty((1,0), float)
        labelPerExample = np.empty((1,0), int)
        for x in unlabeled:#
            #print(x.shape)
         #   count = count + 1
            #print(count)
            predImages = model.predict(x)
            accuPerExample = np.concatenate((accuPerExample,predImages[np.arange(predImages.shape[0]), predImages.argmax(axis=1)]), axis=None)
            labelPerExample = np.concatenate((labelPerExample,predImages.argmax(axis=1)), axis=None)
            
        #print(accuPerExample.shape)
        #print(labelPerExample.shape)
      
        #transferUnlabToAug: moves pseudo-labeled data to labaled data, and delete those from unlabaled data.
        try:
          unlabX_train, augX_train, augy_train = met.transferUnlabToAug(
                                                                  unlabX_train,
                                                                  augX_train,
                                                                  augy_train,
                                                                  accuPerExample,
                                                                  labelPerExample)
        except ValueError:  #raised if `y` is empty.
          pass
          break 
      
        #unlabX_train.shape
        #unlaby_train.shape
        #unlaby_train[exampleToMove]
      
      #writing results on csv
      fn = mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/selftrain.csv'
      
      my_data = genfromtxt(fn, delimiter=',')
      try: 
        my_data.shape[1] #sometimes it is one row then I need a matrix as 1x2
      except IndexError:
        pass
        my_data = np.array([my_data])
          
      data = np.array([my_data[np.argmax(my_data[:,1]),1], AmountTrueLab, int(my_data[np.argmax(my_data[:,1]),0]), np.argmax(my_data[:,1])])
      ## CSV writing
      csv.register_dialect("hashes", delimiter=",")
      f = open(mainDirect+str(AmountTrueLab)+'/'+'selftrainAccuPerTrueLabAnd.csv','a')
      
      with f:
          #fieldnames = ['Accu', 'AmountTrueLab', 'AugDataTrueAndSeudoLabels', Iteration]
          writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
          writer.writerow(data)
      ## CSV writing

def selfTrainingLayerWise(mainDirect,
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
                Epochs):

  for AmountTrueLab in setTrueLabels:
    print(AmountTrueLab)
    #AmountTrueLab = 200
  
    #creating folders to save results and weigths
    try:
        os.makedirs(mainDirect+str(AmountTrueLab))
    except FileExistsError:
          # directory already exists
        pass
    
    for tenTimes in range(0,10,1):#10
      tf.keras.backend.clear_session()
      try:
          os.makedirs(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes))
      except FileExistsError:
            # directory already exists
          pass
      
      #X_train.shape
      #X_test.shape
      
      #unlabeled data
      unlabX_train = unlabeledXtrain
      #unlaby_train = y_train[:50000]
      
      #labeled data
      labX_train = labeledX
      laby_train = labeledy
      
      #amount of labeled data
      datatrain = AmountTrueLab #200,600, AmountTrueLab
      #trainBATCH_SIZE = 32
      #testBATCH_SIZE = 512
      #unlabBATCH_SIZE = 512
      
      features = labX_train.shape[2]
      BUFFER_SIZE = 10000
      LAYER = 3
      UNITS = 512
      
      #It is not within loop
      
      def get_lr_metric(optimizer):
            def lr(y_true, y_pred):
                return optimizer.learning_rate
            return lr
      
      #RNN-LSTM setup
      model = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=labX_train.shape[-2:],dropout=0.0),
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
          tf.keras.layers.LSTM(units=UNITS),
          tf.keras.layers.Dense(features),
          #tf.keras.layers.Dense(28, activation='relu'),
          tf.keras.layers.Dense(10,activation='softmax')])#25 for slmnist    
      
      opt = tf.keras.optimizers.Adam(1e-3)
      lr_metric = get_lr_metric(opt)
      model.compile(optimizer=opt,
                    loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
                    metrics=['acc', lr_metric])
      #model.summary()
      
      #Augmented training examples
      augX_train = labX_train[:datatrain]
      augy_train = laby_train[:datatrain]
      
      cbks = [tf.keras.callbacks.LearningRateScheduler(lambda epoch: (1e-3)/((epoch+1)**(1/2))), #(1e-3)/(epoch+1) #0.001
                #tf.keras.callbacks.TensorBoard(log_dir="logs/{}".format(time())),
                tf.keras.callbacks.EarlyStopping(monitor='loss', mode='min', verbose=1),
                tf.keras.callbacks.CSVLogger(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/'+str(augX_train.shape[0])+'.csv', append=True, separator=',')
                ]#(write_graph=True)] # https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
      
      #RNN-LSTM setup to load initialised weights
      pretrainedModel = tf.keras.models.Sequential([
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True, input_shape=labX_train.shape[-2:],dropout=0.0),
          tf.keras.layers.LSTM(units=UNITS, return_sequences=True),
          tf.keras.layers.LSTM(units=UNITS,return_sequences=True),
          tf.keras.layers.Dense(features) #tested with tanh, sigmoid, linear and the better accu (41.10%) was reached without any func actv.
      ])
      
      #pretrainedModel.summary()
      
      #loading initialised weights
      pretrainedModel.load_weights(pretrainfile+'.h5')
      
      #pretrainedModel.layers[0].get_weights()
      
      model.layers[0].set_weights(pretrainedModel.layers[0].get_weights())
      model.layers[1].set_weights(pretrainedModel.layers[1].get_weights())
      model.layers[2].set_weights(pretrainedModel.layers[2].get_weights())
      model.layers[3].set_weights(pretrainedModel.layers[3].get_weights())

      #Loop (if i >= 8 then numberExamplesPerClass is 1000)
      while (unlabX_train.shape[0] != 0): #LOOP to ADD PSEUDO-LABELS INTO TRUE LABELS unlabX_train.shape[0]
        #print(unlabX_train.shape)
        #augX_train.shape
        #augy_train.shape
        
        unlabeled = tf.data.Dataset.from_tensor_slices((unlabX_train)) # [:datatrain]
        unlabeled = unlabeled.batch(unlabBATCH_SIZE)
        
        train = tf.data.Dataset.from_tensor_slices((augX_train, augy_train)) # [:datatrain]
        train = train.batch(trainBATCH_SIZE).repeat()
        
        test = tf.data.Dataset.from_tensor_slices((X_test, y_test))
        test = test.batch(testBATCH_SIZE).repeat()
        
        #training
        stepsPerEpoch = math.floor(augX_train.shape[0]/trainBATCH_SIZE)
        stepsTest = math.floor(X_test.shape[0]/testBATCH_SIZE)
        model.fit(train,epochs=Epochs, 
                              verbose=0,steps_per_epoch=stepsPerEpoch,  #58 for 60k (training data) entre 1024 (batch zise) = 58.59
                              validation_data=test,
                              validation_steps=stepsTest, #9 for 10k (test data) entre 1024 (batchsize)
                              callbacks=cbks
                  )
        model.save_weights(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/'+str(augX_train.shape[0])+'.h5')
        
        #model.load_weights('/content/drive/My Drive/PhD/Semestre 3/model implementation/LSTM semi-supervised/self-training/weights/'+str(augX_train.shape[0])+'.h5')
        
        #pred = model.predict(unlabeled,steps=19).shape # batch_size=unlabBATCH_SIZE,
        
        #getting the results
        results = model.evaluate(x=test, verbose=1, steps=stepsTest) #batch_size=testBATCH_SIZE,
        amodel = [augX_train.shape[0], round(results[1]*100,2)]
        
        ##writing the results on csv
        csv.register_dialect("hashes", delimiter=",")
        f = open(mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/selftrainLayerWise.csv','a')
        
        with f:
            #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
            writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
            writer.writerow(amodel)
        ## CSV writing
      
        #Getting Accu and labels per example from predictions.
        #count = 0
        accuPerExample = np.empty((1,0), float)
        labelPerExample = np.empty((1,0), int)
        for x in unlabeled:#
            #print(x.shape)
         #   count = count + 1
            #print(count)
            predImages = model.predict(x)
            accuPerExample = np.concatenate((accuPerExample,predImages[np.arange(predImages.shape[0]), predImages.argmax(axis=1)]), axis=None)
            labelPerExample = np.concatenate((labelPerExample,predImages.argmax(axis=1)), axis=None)
            
        #print(accuPerExample.shape)
        #print(labelPerExample.shape)
      
        #transferUnlabToAug: moves pseudo-labeled data to labaled data, and delete those from unlabaled data.
        try:
          unlabX_train, augX_train, augy_train = met.transferUnlabToAug(
                                                                  unlabX_train,
                                                                  augX_train,
                                                                  augy_train,
                                                                  accuPerExample,
                                                                  labelPerExample)
        except ValueError:  #raised if `y` is empty.
          pass
          break 
      
        #unlabX_train.shape
        #unlaby_train.shape
        #unlaby_train[exampleToMove]
      
      #writing results on CSV
      fn = mainDirect+str(AmountTrueLab)+'/'+str(tenTimes)+'/selftrainLayerWise.csv'
      
      my_data = genfromtxt(fn, delimiter=',')
      try: 
        my_data.shape[1] #sometimes it is one row then I need a matrix as 1x2
      except IndexError:
        pass
        my_data = np.array([my_data])
          
      data = np.array([my_data[np.argmax(my_data[:,1]),1], AmountTrueLab, int(my_data[np.argmax(my_data[:,1]),0]), np.argmax(my_data[:,1])])
      ## CSV writing
      csv.register_dialect("hashes", delimiter=",")
      f = open(mainDirect+str(AmountTrueLab)+'/'+'selftrainLayerWiseAccuPerTrueLabAnd.csv','a')
      
      with f:
          #fieldnames = ['Accu', 'AmountTrueLab', 'AugDataTrueAndSeudoLabels', Iteration]
          writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
          writer.writerow(data)
      ## CSV writing

  #Final Results for chart
  results = np.empty((0,3), float)
  for AmountTrueLab in setTrueLabels:
  
    print(AmountTrueLab)
  
    fn = mainDirect+str(AmountTrueLab)+'/selftrainLayerWiseAccuPerTrueLabAnd.csv'
  
    my_data = genfromtxt(fn, delimiter=',')
    try: 
      my_data.shape[1]
    except IndexError:
      pass
      my_data = np.array([my_data])
      
    meanAccu = round(np.mean(my_data[:,0]),2) 
    results = np.concatenate((results,meanAccu), axis=None)
  
  csv.register_dialect("hashes", delimiter=",")
  f = open(mainDirect+'/meanAccuTenTimes.csv','a')
        
  with f:
    #fieldnames = ['layer', 'units', 'epochsCv', 'AccuCv', 'epochsTrain', 'AccuTest']
    writer = csv.writer(f, dialect="hashes")#,fieldnames=fieldnames)
    writer.writerow(results)
