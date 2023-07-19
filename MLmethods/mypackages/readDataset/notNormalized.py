# NotNormalized
# How to build packages https://www.pythonlikeyoumeanit.com/Module5_OddsAndEnds/Modules_and_Packages.html


import pylab as p
import numpy as np
import os
import struct
from random import randrange
#import seaborn as sns
from matplotlib.colors import SymLogNorm
import pandas as pd
from six.moves import cPickle as pickle 

#class notNormalized:

def mnist():
      def load_mnist(path, kind='train'):
          """Load MNIST data from `path`"""
          labels_path = os.path.join(path,'%s-labels-idx1-ubyte' % kind)
          images_path = os.path.join(path,'%s-images-idx3-ubyte' % kind)
          with open(labels_path, 'rb') as lbpath:
              magic, n = struct.unpack('>II',lbpath.read(8))
              labels = np.fromfile(lbpath,dtype=np.uint8)
          with open(images_path, 'rb') as imgpath:
              magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
              images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
              #images = ((images / 255.) - .5) * 2
          return images, labels

      ## loading the data
      X_train, y_train = load_mnist('/home/est1/Edgar/tensorflowPractice/mnist', kind='train')
      #print('Rows: %d, Columns: %d' %(X_train.shape[0],
      #                                X_train.shape[1]))

      X_test, y_test = load_mnist('/home/est1/Edgar/tensorflowPractice/mnist', kind='t10k')
      #print('Rows: %d, Columns: %d' %(X_test.shape[0],
      #                                X_test.shape[1]))

      # With the function above the X_train is normalized in -1 to 1.
      # This normalization is made because behind this is that gradient-based optimization is much more stable.

      # Visualize the normalized data.
      #plt.imshow(X_train[0].reshape((28,28)))

      # Becoming 2D (X_train) to 3D(X_train)
      X_train = X_train[:,:,np.newaxis]
      X_test = X_test[:,:,np.newaxis]

      # Reshaping d(60k,784,1) to d(60k,28,28)
      X_train = X_train.reshape((len(y_train),28,28))
      X_test = X_test.reshape((len(y_test),28,28))
      return X_train, X_test, y_train, y_test

def mnist_c():
        
    X_train = np.load('/home/est1/Edgar/tensorflowPractice/mnist_c/glass_blur/train_images.npy')
    X_test = np.load('/home/est1/Edgar/tensorflowPractice/mnist_c/glass_blur/test_images.npy')
    y_train = np.load('/home/est1/Edgar/tensorflowPractice/mnist_c/glass_blur/train_labels.npy')
    y_test = np.load('/home/est1/Edgar/tensorflowPractice/mnist_c/glass_blur/test_labels.npy')
        
        # Becoming 4D (X_train) to 3D(X_train)
    X_train = X_train[:,:,:,0]
    X_test = X_test[:,:,:,0]
        #X_train.shape
        #X_test.shape
        #y_train.shape
        #y_test.shape

        # Normalize data
        
        # X_train[11,:] = ((X_train[0,:] / 255.) - .5) * 2
        #X_train = ((X_train / 255.) - .5) * 2
        #X_test = ((X_test / 255.) - .5) * 2
        #np.unique(X_train[1000]) asuring normalized data. ok.
        
        # Reshaping d(60k,784,1) to d(60k,28,28)
    X_train = X_train.reshape((len(y_train),28,28))
    X_test = X_test.reshape((len(y_test),28,28))
    return X_train, X_test, y_train, y_test

def notmnist():
        # def loadData(): 
            # for reading also binary mode is important 
      dbfile = open('/home/est1/Edgar/tensorflowPractice/notMNIST/notMNIST.pickle', 'rb') #Useful to train, please, don't remove notMNIST.pickle     
      db = pickle.load(dbfile) 
      for keys in db: 
          #print(keys)#, '=>', db[keys]) 
            
          train_dataset = db['train_dataset']
          train_labels= db['train_labels']
          valid_dataset = db['valid_dataset']
          valid_labels = db['valid_labels']
          test_dataset = db['test_dataset']
          test_labels = db['test_labels']
            
      dbfile.close() 
      
      # Prepare training data
      samples, width, height = train_dataset.shape
      X_train = np.reshape(train_dataset,(samples,width*height))
      y_train = train_labels
        
      # Prepare testing data
      samples, width, height = test_dataset.shape
      X_test = np.reshape(test_dataset,(samples,width*height))
      y_test = test_labels
      
      #Denormalize
      X_train = np.around(((X_train/2)+0.5)*255)
      X_test =  np.around(((X_test/2)+0.5)*255)
      
      #float to int
      X_train = X_train.astype(np.int64)  
      X_test = X_test.astype(np.int64)  

      # Becoming 2D (X_train) to 3D(X_train)
      X_train = X_train[:,:,np.newaxis]
      X_test = X_test[:,:,np.newaxis]
      
      # Reshaping d(60k,784,1) to d(60k,28,28)
      X_train = X_train.reshape((len(y_train),28,28))
      X_test = X_test.reshape((len(y_test),28,28))
      
      return X_train, X_test, y_train, y_test

def fashion():
      def load_mnist(path, kind='train'):
          """Load MNIST data from `path`"""
          labels_path = os.path.join(path,'%s-labels-idx1-ubyte' % kind)
          images_path = os.path.join(path,'%s-images-idx3-ubyte' % kind)
          with open(labels_path, 'rb') as lbpath:
              magic, n = struct.unpack('>II',lbpath.read(8))
              labels = np.fromfile(lbpath,dtype=np.uint8)
          with open(images_path, 'rb') as imgpath:
              magic, num, rows, cols = struct.unpack(">IIII",imgpath.read(16))
              images = np.fromfile(imgpath,dtype=np.uint8).reshape(len(labels), 784)
              #images = ((images / 255.) - .5) * 2
          return images, labels

      ## loading the data
      X_train, y_train = load_mnist('/home/est1/Edgar/tensorflowPractice/mnistFashion', kind='train')
      #print('Rows: %d, Columns: %d' %(X_train.shape[0],
      #                                X_train.shape[1]))

      X_test, y_test = load_mnist('/home/est1/Edgar/tensorflowPractice/mnistFashion', kind='t10k')
      #print('Rows: %d, Columns: %d' %(X_test.shape[0],
      #                                X_test.shape[1]))

      # With the function above the X_train is normalized in -1 to 1.
      # This normalization is made because behind this is that gradient-based optimization is much more stable.

      # Visualize the normalized data.
      #plt.imshow(X_train[0].reshape((28,28)))

      # Becoming 2D (X_train) to 3D(X_train)
      X_train = X_train[:,:,np.newaxis]
      X_test = X_test[:,:,np.newaxis]

      # Reshaping d(60k,784,1) to d(60k,28,28)
      X_train = X_train.reshape((len(y_train),28,28))
      X_test = X_test.reshape((len(y_test),28,28))
      return X_train, X_test, y_train, y_test

def handmnist():
      path = "/home/est1/Edgar/tensorflowPractice/sign-language-mnist/sign_mnist_train/sign_mnist_train.csv"
      df = pd.read_csv(path)
        #df.shape

      #Selecting ten classes only in train part
      df_filtered = df[df['label'] > 9]
      df = df_filtered
      df_filtered = df[df['label'] < 20]
      df = df_filtered
        
        #Train Data version1
      X_train = df.values[:,1:]
      y_train = df.values[:,0]
        
        #Test Data version1
      path = "/home/est1/Edgar/tensorflowPractice/sign-language-mnist/sign_mnist_test/sign_mnist_test.csv"
      df = pd.read_csv(path)
        
      #Selecting ten classes only in test part
      df_filtered = df[df['label'] > 9]
      df = df_filtered
      df_filtered = df[df['label'] < 20]
      df = df_filtered
        
        #df.shape
      X_test = df.values[:,1:]
      y_test = df.values[:,0]
        
        # Becoming from np.df data to np.array and from int to float64
        
      X_train = np.array(X_train).astype(np.int64)
      y_train = np.array(y_train)
        
      X_test = np.array(X_test).astype(np.int64)
      y_test = np.array(y_test)
        
        # Normalize data
        
        # X_train[11,:] = ((X_train[0,:] / 255.) - .5) * 2
        #X_train = ((X_train / 255.) - .5) * 2
        #X_test = ((X_test / 255.) - .5) * 2
        
        # Becoming 2D (X_train) to 3D(X_train)
      X_train = X_train[:,:,np.newaxis]
      X_test = X_test[:,:,np.newaxis]  

        # Reshaping d(60k,784,1) to d(60k,28,28)
      X_train = X_train.reshape((len(y_train),28,28))
      X_test = X_test.reshape((len(y_test),28,28))
      return X_train, X_test, y_train, y_test