#!/usr/bin/env python
# coding: utf-8
# AMD or NVIDIA
try:
    import plaidml.keras
    plaidml.keras.install_backend()
except:
    import keras

import os
import random
from distutils.util import strtobool
from glob import glob
from pathlib import Path

import keras.utils
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sn
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

import constant_log
import cv2

#### ------- Constants -------

Result_Table = {}

#### ------- I. Find Data -------


def Detect_Data_from_directory(path):
    '''
    Input :
        Alldata_Path = directory for all data
    Output:
        Result_Table = check number of subdirectory and update the class
        Subdata_Path = all subdirectory path
    '''
    Alldata_Path = Path(path)
    Subdata_Path = []
    Tmpdata_Path = [x for x in Alldata_Path.iterdir() if x.is_dir()]
    for idx, path in enumerate(Tmpdata_Path):
        Result_Table[idx] = os.path.basename(os.path.normpath(path))
        Subdata_Path.append(path.as_posix())

    print("Class found:  ", Result_Table)
    print("Class data Path:  ", Subdata_Path)
    return Subdata_Path


#### ------- II. Load Data -------


def read_Img_Lbl(subpath):
    """
    Input :
        subpath = list of each class data path 
	Output:
        X : list of All class image data
        y : list of All class image label
        fn : list of All class image filename
    """
    X = []
    y = []
    fn = []
    for idx, single_class_path in enumerate(subpath):
        directory = os.path.abspath(os.path.join(single_class_path))
        bmp_path = glob(os.path.join(directory, "*.bmp"))
        print("Class ", Result_Table[idx], ": ", len(bmp_path),
              " Image Found.")
        for files in bmp_path:
            X.append(files)
            if idx is not None:
                y.append(idx)
            fn.append(os.path.split(files)[1])

    return X, y, fn


#### ------- III. Spilt Data -------
'''
    !!! USING train_test_spilt function.
    Change test_size for 
            the persentage of test data  float(<1) : 0.25 , 0.34  , ...
            the persentage of test data  int       : 250 , 340 , ...
'''


def Spilt_Data_2_train_test(Img, Lbl, num_test):
    combined = list(zip(Img, Lbl))
    random.shuffle(combined)
    Imgdir_random, Lbl_random = [], []
    Imgdir_random[:], Lbl_random[:] = zip(*combined)
    Imgdir_train = Imgdir_random[0:len(Imgdir_random) - num_test]
    Imgdir_test = Imgdir_random[len(Imgdir_random) -
                                num_test:len(Imgdir_random)]
    Lbl_train = Lbl_random[0:len(Lbl_random) - num_test]
    Lbl_test = Lbl_random[len(Lbl_random) - num_test:len(Lbl_random)]

    return Imgdir_train, Imgdir_test, Lbl_train, Lbl_test


#### ------- IV. Adjust Data -------


def Data_Adjust(Img, Lbl, Categorical_check):
    Img = [img_hist_equ(img) for img in Img]
    Img = Normalize_img(Img)
    if Categorical_check is True:
        Lbl = Categorical_label(Lbl)
    return Img, Lbl


def Normalize_img(X):
    # X = np.asarray(X)
    X = X.astype("float32")
    X = X / 255
    return X


def Categorical_label(y):
    # y = np.asarray(y)
    y = keras.utils.to_categorical(y, num_classes=len(Result_Table) + 1)
    return y


def img_hist_equ(img, number_bins=256):
    img_hist, bins = np.histogram(img.flatten(), number_bins, density=True)
    cdf = img_hist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    equ = np.interp(img.flatten(), bins[:-1], cdf)
    return equ.reshape(img.shape)


#### ------- V. Train Model -------


def setModel(model_filename, imgTrain, lblTrain):

    if os.path.isfile(model_filename):
        print("Model \"%s\" found." % model_filename)
        retrain_check = user_yes_no_query("Retraining Model?")
    else:
        print("Model \"%s\" not found: start training." % model_filename)
        print("Training Model \"%s\"......" % model_filename)
        retrain_check = 1

    print("\n")
    my_training_batch_generator = My_Generator(imgTrain, lblTrain, 100)

    if retrain_check == 1:
        print(constant_log.PHASE5_STR_TRAIN_MODEL)
        model = SimpleCNN(200, 200)
        #model.fit(imgTrain, lblTrain, batch_size=10, epochs=50)
        model.fit_generator(
            generator=my_training_batch_generator,
            steps_per_epoch=(12000 / 100),
            epochs=50,
            verbose=1,
            validation_data=None,
            validation_steps=None,
            use_multiprocessing=False)
        '''
        fit_generator(generator,\
                      steps_per_epoch=None,\
                      epochs=1, verbose=1, callbacks=None,\
                      validation_data=None, validation_steps=None,\
                      validation_freq=1, class_weight=None,\
                      max_queue_size=10, workers=1,\
                      use_multiprocessing=False, shuffle=True, initial_epoch=0)
        '''
        model.save(model_filename)

    model = load_model(model_filename)
    return model


def SimpleCNN(sizeX, sizeY):
    '''
        Simple CNN Structure
    '''
    model = Sequential()
    model.add(
        Conv2D(50, (3, 3), activation="relu", input_shape=(sizeX, sizeY, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(
        Conv2D(50, (3, 3), activation="relu", input_shape=(sizeX, sizeY, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(100, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(100, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(len(Result_Table) + 1, activation="softmax"))
    model.compile(
        loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return model


#### ------- VI. Analyzing Testing Data -------


def Analyzing_Test_Predict(lblTrue, lblPred):

    print("\n")
    err_idx = []
    for idx, pred in enumerate(lblPred):
        if pred != lblTrue[idx]:
            err_idx.append(idx)

    for idx, pred in enumerate(lblPred):
        if lblTrue[idx] == pred:
            print("    Test{: >5d}".format(idx + 1), "~~   ", "Real:",
                  Result_Table[lblTrue[idx]], "   Predict:",
                  Result_Table[pred])
        else:
            print("  **Test{: >5d}".format(idx + 1), "~~   ", "Real:",
                  Result_Table[lblTrue[idx]], "   Predict:",
                  Result_Table[pred])

    PrintConfusionMatrix(lblTrue, lblPred)

    print("F1_score:  \n", f1_score(lblTrue, lblPred, average=None))

    return err_idx


def PrintConfusionMatrix(test_correct_lb, test_pred_lb):
    pd.options.display.float_format = '${:20,0f}'.format
    df_cm = pd.DataFrame(confusion_matrix(test_correct_lb, test_pred_lb),\
                     index = [ Result_Table[i] for i in Result_Table ],\
                     columns = [ Result_Table[i] for i in Result_Table ])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    print("\nShow and Save Confusion_Matrix.jpg")
    plt.savefig('confusion_matrix.jpg')
    print(confusion_matrix(test_correct_lb, test_pred_lb))


#### -------- Optional Function------


class My_Generator(keras.utils.Sequence):
    def __init__(self, image_filenames, labels, batch_size):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[idx * self.batch_size:(idx + 1) *
                                       self.batch_size]
        batch_y = self.labels[idx * self.batch_size:(idx + 1) *
                              self.batch_size]
        return keras.utils.normalize(np.array([imread(file_name) for file_name in batch_x])),\
               keras.utils.to_categorical(np.array(batch_y), num_classes = len(Result_Table)+1)


def user_yes_no_query(question):
    print('%s [y/n]' % question)
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            print('Please respond with \'y\' or \'n\'.\n')


#### --------------------------------


def adjustData(X, y):
    X = np.asarray(X)
    X = X.astype("float32")
    X = X / 255
    if y is not None:
        y = np.asarray(y)
        y = np_utils.to_categorical(y, num_classes=5)
    return X, y


def runCNN(model_file):

    print(constant_log.PHASE1_STR_DETECT_FILE)
    EveryClassPath = Detect_Data_from_directory("data/")

    print(constant_log.PHASE2_STR_LOAD_IMAGE)
    ImgData, LblData, ImgName = read_Img_Lbl(EveryClassPath)
    """
    print(constant_log.PHASE3_STR_SPILT_DATA)
    imgTrain, imgTest, lblTrain, lblTest = Spilt_Data_2_train_test(
        ImgData, LblData, 3000)

    
    '''
    print(constant_log.PHASE4_STR_ADJUST_DATA)
    imgTrain, lblTrain = Data_Adjust(imgTrain, lblTrain ,True)
    imgTest, lblTest = Data_Adjust(imgTest, lblTest ,False)
    '''

    print(constant_log.PHASE5_STR_LOAD_MODEL)
    model = setModel(model_file, imgTrain, lblTrain)

    print(constant_log.PHASE6_STR_RUN_PRED)
    lblPred = model.predict_classes(imgTest)

    print(constant_log.PHASE6_STR_ANALZYE_PRED)
    Analyzing_Test_Predict(lblTest, lblPred)
    """
    train_data, train_label, test_data, test_label = train_test_split(
        ImgData, LblData, test_size=0.25, random_state=42)
    model = load_model("model.h5")
    adj_test_data, adj_test_label = adjustData(test_data, test_label)
    score = model.evaluate(adj_test_data, adj_test_label)
    print(score)


if __name__ == "__main__":
    #os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    runCNN("model.h5")
