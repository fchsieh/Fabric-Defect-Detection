#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob

import matplotlib.pyplot as plt
import numpy as np

try:
    import plaidml.keras
    plaidml.keras.install_backend()
except:
    import keras
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import cv2


def adjustData(X, y):
    X = np.asarray(X)
    X = X.astype("float32")
    X = X / 255
    if y is not None:
        y = np.asarray(y)
        y = np_utils.to_categorical(y, num_classes=4)
    return X, y


def SimpleCNN(sizeX, sizeY):
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
    model.add(Dense(4, activation="softmax"))
    model.compile(
        loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return model


def readImageAndLabel(dirname, idx=None, X=None, y=None):
    if X is None:
        X = []
    if y is None:
        y = []
    fn = []
    directory = os.path.abspath(os.path.join(dirname))
    path = glob(os.path.join(directory, "*.bmp"))
    for files in path:
        X.append(img_to_array(load_img(files)))
        if idx is not None:
            y.append(idx)
        fn.append(os.path.split(files)[1])
    return X, y, fn


def printMispred(X, y, model):
    # create result dir if not exists
    if not os.path.exists("false_pred"):
        os.makedirs("false_pred")
        os.makedirs("false_pred/pred_normal")
        os.makedirs("false_pred/pred_spiderweb")
        os.makedirs("false_pred/pred_stain")
    adj_X, _ = adjustData(X, y)
    y_pred = model.predict_classes(adj_X)
    err_idx = []
    for idx, pred in enumerate(y_pred):
        if pred != y[idx]:
            err_idx.append(idx)

    for i in range(len(err_idx)):
        if y[err_idx[i]] == 0:
            true = "normal"
        elif y[err_idx[i]] == 1:
            true = "spiderweb"
        elif y[err_idx[i]] == 2:
            true = "stain"
        if y_pred[err_idx[i]] == 0:
            pred = "normal"
        elif y_pred[err_idx[i]] == 1:
            pred = "spiderweb"
        elif y_pred[err_idx[i]] == 2:
            pred = "stain"

        cv2.imwrite(
            "false_pred/pred_%s/true-%s-pred-%s-%d.bmp" %
            (pred, true, pred, err_idx[i]), X[err_idx[i]])
    print(confusion_matrix(y, y_pred))


def evaluate():
    # normal: 0
    # spiderweb: 1
    # stain: 2
    X = []
    y = []
    X, y, fn_normal = readImageAndLabel("data/正常", 0, X, y)
    X, y, fn_spiderweb = readImageAndLabel("data/蛛網", 1, X, y)
    X, y, fn_stain = readImageAndLabel("data/碰污", 2, X, y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    adj_X_train, adj_y_train = adjustData(X_train, y_train)
    adj_X_test, adj_y_test, = adjustData(X_test, y_test)
    try:
        _ = open("model.h5")
        print("Load model from model.h5")
    except:
        # model does not exists
        # 200x200 px
        model = SimpleCNN(200, 200)
        model.fit(adj_X_train, adj_y_train, batch_size=10, epochs=50)
        model.save("model.h5")
    model = load_model("model.h5")
    score = model.evaluate(adj_X_test, adj_y_test, batch_size=10)
    print("Accuracy:", score[1])
    # print mispredicted images
    filename = fn_normal + fn_spiderweb
    printMispred(X, y, model)


def eval_with_equ():
    X = []
    y = []
    X, y, _ = readImageAndLabel("data/正常", 0, X, y)
    X, y, _ = readImageAndLabel("data/蛛網", 1, X, y)
    X, y, _ = readImageAndLabel("data/碰污", 2, X, y)
    X = [img_hist_equ(img) for img in X]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    adj_X_train, adj_y_train = adjustData(X_train, y_train)
    adj_X_test, adj_y_test, = adjustData(X_test, y_test)

    try:
        _ = open("model_equ.h5")
        print("Load model from model_equ.h5")
    except:
        # model does not exists
        # 200x200 px
        model = SimpleCNN(200, 200)
        model.fit(adj_X_train, adj_y_train, batch_size=10, epochs=50)
        model.save("model_equ.h5")
    model = load_model("model_equ.h5")
    score = model.evaluate(adj_X_test, adj_y_test, batch_size=10)
    print("Accuracy:", score[1])
    # print mispredicted images
    printMispred(X, y, model)


def loadModel(model_filename):
    # 正常: 0
    # 蛛網: 1
    try:
        _ = open(model_filename)
        print("Load model from \"%s\"." % model_filename)
    except:
        print("\"%s\" not found: start training." % model_filename)
        X = []
        y = []
        X, y, _ = readImageAndLabel("data/正常", 0, X, y)
        X, y, _ = readImageAndLabel("data/蛛網", 1, X, y)
        X, y, _ = readImageAndLabel("data/碰污", 2, X, y)
        adj_X, adj_y = adjustData(X, y)
        model = SimpleCNN(200, 200)
        model.fit(adj_X, adj_y, batch_size=10, epochs=50)
        model.save(model_filename)
    model = load_model(model_filename)
    return model


def img_hist_equ(img, number_bins=256):
    img_hist, bins = np.histogram(img.flatten(), number_bins, density=True)
    cdf = img_hist.cumsum()
    cdf = 255 * cdf / cdf[-1]
    equ = np.interp(img.flatten(), bins[:-1], cdf)
    return equ.reshape(img.shape)


def classify(model_file):
    """
    Do image classification.

    Testing data:
        put testing data under /test/ directory
    """
    # load model
    model = loadModel(model_file)
    test_data, _, filename = readImageAndLabel("test")
    test_data, _ = adjustData(test_data, None)

    y_pred = model.predict_classes(test_data)
    y_pred_result = []
    for i in y_pred:
        if i == 1:
            y_pred_result.append("蛛網")
        elif i == 2:
            y_pred_result.append("碰污")
        elif i == 0:
            y_pred_result.append("正常")

    result = dict(zip(filename, y_pred_result))
    for res in result:
        print(res, ":", result[res])


if __name__ == "__main__":
    classify("model.h5")
    #evaluate()
    #eval_with_equ()
