import os
import numpy as np
from keras.models import *
from keras.layers import Input, Cropping2D, concatenate, Concatenate, merge, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D, add, Dense
from keras.optimizers import *
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, ReduceLROnPlateau
from keras_applications import vgg16
from keras import backend as kerasB
import keras as KR
from data import *
from conv2d_LC_layer import Conv2D_LC

#import test_predict
class myGESUnet(object):

   def __init__(self, img_rows = 128, img_cols = 128):


      self.img_rows = img_rows
      self.img_cols = img_cols

   def load_data(self):

      mydata = dataProcess(self.img_rows, self.img_cols)
      mydata.create_train_data() 
      mydata.create_test_data()

      #myAugdata = myAugmentation(self)
      #myAugdata.Augmentation()
      #myAugdata.splitMerge()
      imgs_train, imgs_mask_train = mydata.load_train_data()
      imgs_test = mydata.load_test_data()

      imgs_train /= 255
      imgs_test /= 255
      return imgs_train, imgs_mask_train, imgs_test


   def get_gesunet(self):
      # The first U-net:
      #The fırst net is based on convolution with predefined filters. Number of filters is limited with sample size

      inputs = Input((self.img_rows, self.img_cols, 1), name='first_data')
      #data_format = 'channels_last'
      #num_classes = 2

      S = 9
      K = (3, 3)
      nf=64
      conv1 = Conv2D_LC(num_filters=nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                    kernel_initializer='he_normal')(inputs)
      conv1 = Conv2D_LC(num_filters=nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                        kernel_initializer='he_normal')(conv1)
      conv1 = Conv2D_LC(num_filters=nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv1)
      pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

      conv2 = Conv2D_LC(num_filters=2*nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                        kernel_initializer='he_normal')(pool1)
      conv2 = Conv2D_LC(num_filters=2*nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv2)
      conv2 = Conv2D_LC(num_filters=2*nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv2)
      pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

      conv3 = Conv2D_LC(num_filters=4*nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool2)
      conv3 = Conv2D_LC(num_filters=4*nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                        kernel_initializer='he_normal')(conv3)
      conv3 = Conv2D_LC(num_filters=4*nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv3)
      pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

      conv4 = Conv2D_LC(num_filters=8*nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                    kernel_initializer='he_normal')(pool3)
      conv4 = Conv2D_LC(num_filters=8*nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                        kernel_initializer='he_normal')(conv4)
      conv4 = Conv2D_LC(num_filters=8*nf, kernel_size=K, sample_size=S, activation='relu', padding='same',
                    kernel_initializer='he_normal')(conv4)

      up7 = Conv2D(4*nf, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
         UpSampling2D(size=(2, 2))(conv4))
      merge7 = concatenate([conv3, up7], axis=3)
      conv7 = Conv2D(4*nf, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
      conv7 = Conv2D(4*nf, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

      up8 = Conv2D(2*nf, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
         UpSampling2D(size=(2, 2))(conv7))
      merge8 = concatenate([conv2, up8], axis=3)
      conv8 = Conv2D(2*nf, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
      conv8 = Conv2D(2*nf, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

      up9 = Conv2D(nf, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
         UpSampling2D(size=(2, 2))(conv8))
      merge9 = concatenate([conv1, up9], axis=3)
      conv9 = Conv2D(nf, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
      conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
      #conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
      #conv9 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
      #conv9 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
      conv9 = Conv2D(1, 3, activation='sigmoid', padding='same', kernel_initializer='he_normal')(conv9)

      inputsT = concatenate([inputs, concatenate([inputs, conv9], axis=3)], axis=3)

      modelA = Model(inputs=inputs, output=inputsT)

      inputs2 = Input(shape=(128, 128, 3), name='Sec_data')
      p2 = Input(shape=(4, 4, 1024))

      vgg_conv = vgg16.VGG16(input_tensor=inputs2,
                        weights='imagenet',
                        include_top=False,
                        input_shape=(128, 128, 3),
                        classes=2)

      # vgg_conv = vgg16.VGG16(input_tensor= imgs_train, weights='imagenet', include_top=False)
      #imgs_train = vgg_conv.predict(imgs_train)

      x1 = vgg_conv.get_layer('block1_conv2').output
      x2 = vgg_conv.get_layer('block2_conv2').output
      x3 = vgg_conv.get_layer('block3_conv3').output
      x4 = vgg_conv.get_layer('block4_conv3').output
      x5 = vgg_conv.get_layer('block5_pool').output
      x6 = vgg_conv.get_layer('block5_conv3').output

      conv5t = Conv2D(2048, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x5)
      conv5t = Conv2D(2048, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5t)
      

      
      up6t = Conv2D(1024, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
         UpSampling2D(size=(2, 2))(conv5t))
      
      merge6t = concatenate([x6, up6t], axis=3)
      conv6t = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6t)
      conv6t = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6t)
      

      up7t = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
         UpSampling2D(size=(2, 2))(conv6t))
      

      merge7t = concatenate([x4, up7t], axis=3)
      conv7t = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7t)
      conv7t = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7t)
      

      up8t = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
         UpSampling2D(size=(2, 2))(conv7t))
      
      merge8t = concatenate([x3, up8t], axis=3)
      conv8t = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8t)
      conv8t = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8t)
      

      up9t = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
         UpSampling2D(size=(2, 2))(conv8t))
      merge9t = concatenate([x2, up9t], axis=3)
      conv9t = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9t)
      conv9t = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9t)
      

      up10t = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
         UpSampling2D(size=(2, 2))(conv9t))
      merge10t = concatenate([x1, up10t], axis=3)
      conv10t = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge10t)
      conv10t = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10t)
      conv10t = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10t)
      conv10t = Conv2D(1, 1, activation='sigmoid')(conv10t)

      modelB = Model(inputs=inputT, output=conv10t)


      Out1= modelA(inputs)
      Out = modelB(Out1)
      modelC = Model(inputs=inputs, output=Out)


      modelC.compile(optimizer=Adam(lr=0.000025), loss="mean_squared_error", metrics=['accuracy'])

      return modelC

   def train(self):

      print("loading data")
      imgs_train, imgs_mask_train, imgs_test = self.load_data()
      print('train data size1:', imgs_train.shape)

      print("loading data done")
      model = self.get_gesunet()

      model_checkpoint = ModelCheckpoint('Model_GESU.hdf5', monitor='loss',verbose=1, save_best_only=True)
      print('Fitting model...')
      #class weights optional, if planing to use update the array size 
      #class_weights = np.zeros((16384,2))
      #class_weights[:,0] += 1
      #class_weights[:,1] += 50


      history = model.fit(imgs_train, imgs_mask_train, batch_size=10, verbose=2, nb_epoch=80, validation_split=0.2, shuffle=True, callbacks=[model_checkpoint])

      print('predict test data')
      print(history.history.keys())
      imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)
      np.save('../results/imgs_mask_test.npy', imgs_mask_test)

   def save_img(self):

      print("array to image")
      imgs = np.load('imgs_mask_test.npy')
      for i in range(imgs.shape[0]):
         img = imgs[i]
         img = array_to_img(img)
         img.save("../results/%d.jpg"%(i))


if __name__ == '__main__':
   kerasB.clear_session()
   GESU_net = myGESUnet()
   GESU_net.load_data()
   GESU_net.train()
   #test_predict()

