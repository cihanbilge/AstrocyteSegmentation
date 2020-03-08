
from GESU_net import *
import os
import keras.backend as K
import tensorflow as tf
from keras.models import Model


from data import *
import time
mydata = dataProcess(128,128)
t = time.time()
path='../data/tests_results'

names = mydata.create_test_data()
imgs_test = mydata.load_test_data()
myunet = myUnet()

model = myunet.get_gesunet()

model.load_weights('Modal_GESU.hdf5')

imgs_mask_test = model.predict(imgs_test, batch_size=1, verbose=1)


for i in range(1,imgs_mask_test.shape[0]):
    img = imgs_mask_test[i]
    print(np.max(img))
    img /= np.max(img)
    #if (img[64, 64]<0.5):
    #    img = np.zeros((128,128,1), dtype=np.uint8)
    img = array_to_img(img)

    img.save("../data/tests_results_processed/%d.tif"%names[i])

myAugdata = myAugmentation()
myAugdata.connComp()
elapsed = time.time() - t
print(elapsed)
