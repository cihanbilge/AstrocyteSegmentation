#copied from https://github.com/zhixuhao/unet and edited by Cihan Bilge Kayasandik

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import os
import glob
import cv2
from PIL import Image
#import matplotlib.pyplot as plt


# from libtiff import TIFF

class myAugmentation(object):
    """
    A class used to augmentate image
    Secondly, use keras preprocessing to augmelsdhfpdspfhntate image
    Finally, seperate augmentated image apart into train image and label
    """

    def __init__(self, train_path="/Users/cihanbilgekayasandik/PycharmProjects/untitled11/data/train/image",
                 label_path="/home/bkayasandik/PycharmProjects/untitled11/data/train/label2",
                 merge_path="/home/bkayasandik/PycharmProjects/untitled11/data/train/merge",
                 aug_merge_path="/home/bkayasandik/PycharmProjects/untitled11/data/train/aug_merge",
                 aug_train_path="/home/bkayasandik/PycharmProjects/untitled11/data/train/image",
                 #im_path = "/home/bkayasandik/PycharmProjects/untitled11/data/train/image",
                 aug_label_path="/home/bkayasandik/PycharmProjects/untitled11/data/train/label2", img_type="tif"):

        """
        Using glob to get all .img_type form path
        """
        train_path = "/home/bkayasandik/PycharmProjects/untitled11/data/train/image"
        im_path = "/home/bkayasandik/PycharmProjects/untitled11/data/train/image"
        test_path = "/home/bkayasandik/PycharmProjects/untitled11/data/train/tests"
        merge_path_test = "/home/bkayasandik/PycharmProjects/untitled11/data/train/merge"
        label_test_path = "/home/bkayasandik/PycharmProjects/untitled/results"


        self.train_imgs = glob.glob(train_path + "/*." + img_type)
        self.im_imgs = glob.glob(im_path + "/*." + img_type)
        self.label_imgs = glob.glob(label_path + "/*." + img_type)
        self.test_imgs = glob.glob(test_path + "/*." + img_type)
        self.label_test_imgs = glob.glob(label_test_path + "/*." + "jpg")
        self.train_path = train_path
        self.im_path = im_path
        self.test_path = test_path
        self.label_path = label_path
        self.label_test_path = label_test_path
        self.merge_path = merge_path
        self.merge_path_test = merge_path_test
        self.img_type = img_type
        self.aug_merge_path = aug_merge_path
        self.aug_train_path = aug_train_path
        self.aug_label_path = aug_label_path

        self.datagen = ImageDataGenerator(
            rotation_range=0.2,
            width_shift_range=0.05,
            height_shift_range=0.05,
            shear_range=0.05,
            zoom_range=0.05,
            horizontal_flip=True,
            fill_mode='nearest')

    def Augmentation(self):

        """
        Start augmentation.....
        """
        trains = self.train_imgs
        ims = self.im_imgs
        labels = self.label_imgs
        path_train = self.train_path
        path_label = self.label_path
        path_merge = self.merge_path
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        if len(trains) != len(labels) or len(trains) == 0 or len(trains) == 0:
            # rint "trains can't match labels"
            return 0
        for i in range(1, len(trains)):
            # print trains[i]
            img_t = load_img(trains[i])
            img_i = load_img(ims[i])
            img_l = load_img(labels[i])
            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)
            x_i = img_to_array(img_i)
            x_t[:, :, 2] = x_l[:, :, 0]
            x_t[:, :, 1] = x_i[:, :, 0]
            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge + "/" + str(i) + "." + imgtype)
            img = x_t
            img = img.reshape((1,) + img.shape)
            savedir = path_aug_merge + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            # files = np.random.choice(62, 10, replace=False)
            # for i in range(1,10):
            self.doAugmentate(img, savedir, str(i))

    def OnlyMerge_test(self):

        """
        Start augmentation.....
        """
        test = self.test_imgs
        labels = self.label_test_imgs
        path_test = self.test_path
        path_label = self.label_test_path
        path_merge_test = self.merge_path_test
        imgtype = self.img_type
        path_aug_merge = self.aug_merge_path
        if len(test) != len(labels) or len(test) == 0 or len(labels) == 0:
            # print "tests can't match labels"
            return 0
        for i in range(0, len(test)):
            tname = test[i]
            midname = tname[tname.rindex("/") + 1:]
            filename, file_extension = os.path.splitext(midname)
            # print trains[i]
            img_t = load_img(test[i])
            img_l = load_img(path_label + "/" + filename + ".tif")
            x_t = img_to_array(img_t)
            x_l = img_to_array(img_l)
            # x_l = imgfinals[i]
            x_t[:, :, 2] = x_l[:, :, 0]
            img_tmp = array_to_img(x_t)
            img_tmp.save(path_merge_test + "/" + filename + "." + imgtype)
        # img = x_t
        # img = img.reshape((1,) + img.shape)
        # savedir = path_aug_merge + "/" + str(i)
        # if not os.path.lexists(savedir):
        #	os.mkdir(savedir)
        # files = np.random.choice(62, 10, replace=False)
        # for i in range(1,10):
        # self.doAugmentate(img, savedir, str(i))

    def connComp(self):
        imgall = self.label_test_imgs
        imgfinals = np.ndarray((len(imgall), 128, 128, 1), dtype=np.uint8)
        for i in range(0, len(imgall)):
            im1 = imgall[i]
            midname = im1[im1.rindex("/") + 1:]
            filename, file_extension = os.path.splitext(midname)
            img = cv2.imread(im1, 0)
            # img = cv2.imread('noisy2.png',0)
            # print(img.type)
            th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)[1]
            # img = img_to_array(img)
            # img = img.astype('float32')
            # img[img <= 128] = 0
            # img[img > 128] = 1
            # img = array_to_img(img)
            # img = img_to_array
            # img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)[1]  # ensure binary
            retval, labels = cv2.connectedComponents(th1)

            ##################################################
            # ENLARGEMENT
            ##################################################
            sorted_labels = labels.ravel()
            sorted_labels = np.sort(sorted_labels)

            maxPixel = 150  # eliminate elements with less than maxPixel

            # detect how often an element occurs
            i = 0
            counter = 0

            counterlist = [0] * retval

            while i < len(sorted_labels):
                if sorted_labels[i] == counter:
                    counterlist[counter] = counterlist[counter] + 1
                else:
                    counter = counter + 1
                    i = i - 1

                i = i + 1

            # delete small pixel values
            i = 0
            while i < len(counterlist):
                if counterlist[i] < maxPixel:
                    counterlist[i] = 0
                i = i + 1

            i = 0
            counterlisthelper = []
            while i < len(counterlist):
                if counterlist[i] == 0:
                    counterlisthelper.append(i)
                i = i + 1

            i = 0
            j = 0
            k = 0
            while k < len(counterlisthelper):
                while i < labels.shape[0]:
                    while j < labels.shape[1]:
                        if labels[i, j] == counterlisthelper[k]:
                            labels[i, j] = 0
                        else:
                            labels[i, j] = labels[i, j]
                        j = j + 1
                    j = 0
                    i = i + 1
                i = 0
                j = 0
                k = k + 1

            ##################################################
            ##################################################

            # Map component labels to hue val

            if np.max(labels) == 0:
                label_hue = np.uint8(179 * labels)

            else:
                label_hue = np.uint8(179 * labels / np.max(labels))
                mycenterlabel = label_hue[64, 64]
                label_hue[label_hue != mycenterlabel] = 0
                label_hue[label_hue == mycenterlabel] = 255

            # print(label_hue.shape)

            # cv2.imshow("Input", label_hue)
            # im = np.array(128,128,1)
            # im[:,:,0] = label_hue
            blank_ch = 255 * np.ones_like(label_hue)
            labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

            # cvt to BGR for display
            # labeled_img = cv2.cvtColor(labeled_img, cv2.COLOR_HSV2BGR)

            # set bg label to black
            labeled_img[label_hue == 0] = 0
            labeled_img = Image.fromarray(labeled_img)
            # imgfinals[i] = img_to_array(label_hue)o
            labeled_img.save("/home/bkayasandik/PycharmProjects/untitled11/data/train/tests_results/concomp/%s.tif" % filename)
        # np.save(self.test_path + '/imgs_test_conncomp.npy', imgfinals)
        # name = '/Users/cihanbilgekayasandik/PycharmProjects/untitled/results/%s' % filename
        # np.save(name,labeled_img)

        # return imgfinals

    def doAugmentate(self, img, save_to_dir, save_prefix, batch_size=1, save_format='tif', imgnum=40):


        """
        augmentate one image
        """
        datagen = self.datagen
        i = 0
        for batch in datagen.flow(img,
                                  batch_size=batch_size,
                                  save_to_dir=save_to_dir,
                                  save_prefix=save_prefix,
                                  save_format=save_format):
            i += 1
            if i > imgnum:
                break

    def splitMerge(self):

        """
        split merged image apart
        """
        path_merge = self.aug_merge_path
        path_train = self.aug_train_path
        path_im = self.im_path
        path_label = self.aug_label_path
        self.slices = 117  # added by cihan
        print(0)
        for i in range(1, self.slices):
            path = path_merge + "/" + str(i)
            train_imgs = glob.glob(path + "/*." + self.img_type)
            savedir = path_train  # + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            savedir = path_label  # + "/" + str(i)
            if not os.path.lexists(savedir):
                os.mkdir(savedir)
            for imgname in train_imgs:
                print(1)
                midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + self.img_type)]
                img = cv2.imread(imgname)
                img_train = img[:, :, 2]  # cv2 read image rgb->bgr
                img_label = img[:, :, 0]
                img_im = img[:, :, 1]
                cv2.imwrite(path_train + "/" + midname + "_train" + "." + self.img_type, img_train)
                cv2.imwrite(path_im + "/" + midname + "_train" + "." + self.img_type, img_im)
                # print path_train+"/"+str(i)+"/"+midname+"_train"+"."+self.img_type
                cv2.imwrite(path_label + "/" + midname + "_train" + "." + self.img_type, img_label)

    def splitTransform(self):

        """
        split perspective transform images
        """
        # path_merge = "transform"
        # path_train = "transform/data/"
        # path_label = "transform/label/"
        path_merge = "deform/deform_norm2"
        path_train = "deform/train/"
        path_label = "deform/label/"
        train_imgs = glob.glob(path_merge + "/*." + self.img_type)
        for imgname in train_imgs:
            midname = imgname[imgname.rindex("/") + 1:imgname.rindex("." + self.img_type)]
            img = cv2.imread(imgname)
            img_train = img[:, :, 2]  # cv2 read image rgb->bgr
            img_label = img[:, :, 0]
            cv2.imwrite(path_train + midname + "." + self.img_type, img_train)
            cv2.imwrite(path_label + midname + "." + self.img_type, img_label)


class dataProcess(object):

    def __init__(self, out_rows, out_cols, data_path="/home/bkayasandik/PycharmProjects/untitled11/data/train/image_core",
                 data_path2="/home/bkayasandik/PycharmProjects/untitled11/data/train/image_core",
                 label_path="/home/bkayasandik/PycharmProjects/untitled11/data/train/label2_core",
                 test_path="/home/bkayasandik/PycharmProjects/untitled11/data/train/tests",
                 npy_path="/home/bkayasandik/PycharmProjects/untitled11/npydata", img_type="tif"):

        """


        """

        self.out_rows = out_rows
        self.out_cols = out_cols
        self.data_path = data_path
        self.data_path2 = data_path2
        self.label_path = label_path
        self.img_type = img_type
        self.test_path = test_path
        self.npy_path = npy_path

    def create_train_data(self):
        i = 0
        print('-' * 30)
        print('Creating training images...')
        print('-' * 30)
        imgs = glob.glob(self.data_path + "/*." + self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        imglabels = np.ndarray((len(imgs), self.out_rows, self.out_cols, 1), dtype=np.uint8)
        for imgname in imgs:
            print(i)
            midname = imgname[imgname.rindex("/") + 1:]
            img = load_img(self.data_path + "/" + midname, grayscale=True)
            img2 = load_img(self.data_path2 + "/" + midname, grayscale=True)
            #plt.show(img)
            label = load_img(self.label_path + "/" + midname, grayscale=True)
            img = img_to_array(img)
            #img2 = img_to_array(img2)
            #img_try = array_to_img(img)
            #np.save(self.data_path + "/88888", img_try)
            #img = np.dstack((img, img, img))
            label = img_to_array(label)
            # img = cv2.imread(self.data_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            # label = cv2.imread(self.label_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            # img = np.array([img])
            # label = np.array([label])
            #imgdatas[i,:,:,0] = img[:,:,0]
            #imgdatas[i, :, :, 1] = img[:,:,0]
            #imgdatas[i, :, :, 2] = img[:,:,0]
            imglabels[i,:,:,:] = label
            imgdatas[i] = img
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, len(imgs)))
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_train.npy', imgdatas)
        np.save(self.npy_path + '/imgs_mask_train.npy', imglabels)
        print('Saving to .npy files done.')

    def create_test_data(self):
        i = 0
        print('-' * 30)
        print('Creating test images...')
        print('-' * 30)
        imgs = glob.glob(self.test_path + "/*." + self.img_type)
        print(len(imgs))
        imgdatas = np.ndarray((len(imgs), self.out_rows, self.out_cols, 3), dtype=np.uint8)


        imgdatas_names = np.ndarray((len(imgs), 1), dtype=np.int)
        for imgname in imgs:
            midname = imgname[imgname.rindex("/") + 1:]
            img = load_img(self.test_path + "/" + midname, grayscale=True)
            #img2 = load_img(self.data_path2 + "/" + midname, grayscale=True)
            img = img_to_array(img)
            #img2 = img_to_array(img2)
            # img = cv2.imread(self.test_path + "/" + midname,cv2.IMREAD_GRAYSCALE)
            # img = np.array([img])
            imgdatas[i, :, :, 0] = img[:, :, 0]
            imgdatas[i, :, :, 1] = img[:, :, 0]
            imgdatas[i, :, :, 2] = img[:, :, 0]
            #imgdatas[i] = img
            filename, file_extension = os.path.splitext(midname)
            imgdatas_names[i] = filename
            i += 1
        print('loading done')
        np.save(self.npy_path + '/imgs_test.npy', imgdatas)
        print('Saving to imgs_test.npy files done.')
        return imgdatas_names

    def load_train_data(self):
        print('-' * 30)
        print('load train images...')
        print('-' * 30)
        imgs_train = np.load(self.npy_path + "/imgs_train.npy")
        imgs_mask_train = np.load(self.npy_path + "/imgs_mask_train.npy")
        imgs_train = imgs_train.astype('float32')
        imgs_mask_train = imgs_mask_train.astype('float32')
        imgs_train /= 255
        # mean = imgs_train.mean(axis = 0)
        # imgs_train -= mean
        imgs_mask_train /= 255
        imgs_mask_train[imgs_mask_train > 0.5] = 1
        imgs_mask_train[imgs_mask_train <= 0.5] = 0
        return imgs_train, imgs_mask_train

    def load_test_data(self):
        print('-' * 30)
        print('load test images...')
        print('-' * 30)
        imgs_test = np.load(self.npy_path + "/imgs_test.npy")
        imgs_test = imgs_test.astype('float32')
        imgs_test /= 255
        # mean = imgs_test.mean(axis = 0)
        # imgs_test -= mean
        return imgs_test


if __name__ == "__main__":
     aug = myAugmentation()
     aug.Augmentation()
     aug.splitMerge()
     mydata = dataProcess(128, 128)
     mydata.create_train_data()



    #mydata.create_test_data()
# imgs_train,ccccc = mydata.load_train_data()
# print imgs_train.shape,imgs_mask_train.shape
