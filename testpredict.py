#from visual3_formodal4_reducelr import *
from visual3_formodal333 import *

#import numpy as np
#from ternausLike import *
import os
#from lates_up import *
import keras.backend as K
import tensorflow as tf
#tf.enable_eager_execution()
from keras.models import Model


from data import *
import time
mydata = dataProcess(128,128)
t = time.time()
path='/home/bkayasandik/PycharmProjects/untitled11/data/train/tests_results_brainstem_formodel2'

names = mydata.create_test_data()
imgs_test = mydata.load_test_data()
myunet = myUnet()

model = myunet.get_unet()
print(model.summary())

model.load_weights('unet_forModel_real4again2lr_againlarger2.hdf5')

imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)


for i in range(1,imgs_mask_test.shape[0]):
    img = imgs_mask_test[i]
    print(np.max(img))
    img /= np.max(img)
    #if (img[64, 64]<0.5):
    #    img = np.zeros((128,128,1), dtype=np.uint8)
    img = array_to_img(img)

    img.save("/home/bkayasandik/PycharmProjects/untitled11/data/train/tests_results_brainstem_formodel2/%d.tif"%names[i])

myAugdata = myAugmentation()
myAugdata.connComp()
elapsed = time.time() - t
print(elapsed)