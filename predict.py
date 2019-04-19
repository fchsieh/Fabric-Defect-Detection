#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob

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


def readImageAndLabel(dirname, idx, X, y):
    directory = os.path.abspath(os.path.join("data", dirname))
    path = glob(os.path.join(directory, "*.bmp"))
    for files in path:
        X.append(img_to_array(load_img(files)))
        y.append(idx)
    return X, y


def readImage(dirname):
    X = []
    directory = os.path.abspath(os.path.join(dirname))
    path = glob(os.path.join(directory, "*.bmp"))
    for files in path:
        X.append(img_to_array(load_img(files)))
    return X


def printMispred(X, y, model):
    # create result dir if not exists
    if not os.path.exists("false_pred"):
        os.makedirs("false_pred")
        os.makedirs("false_pred/pred_normal")
        os.makedirs("false_pred/pred_spiderweb")
    adj_X, _ = adjustData(X, y)
    y_pred = model.predict_classes(adj_X)
    err_idx = []
    for idx, pred in enumerate(y_pred):
        if pred != y[idx]:
            err_idx.append(idx)

    for i in range(len(err_idx)):
        true = "normal" if not y[err_idx[i]] else "spiderweb"
        pred = "normal" if not y_pred[err_idx[i]] else "spiderweb"
        if pred == "normal":
            cv2.imwrite(
                "./false_pred/pred_normal/true_%s_pred_%s-%d.png" %
                (true, pred, err_idx[i]), X[err_idx[i]])
        elif pred == "spiderweb":
            cv2.imwrite(
                "./false_pred/pred_spiderweb/true_%s_pred_%s-%d.png" %
                (true, pred, err_idx[i]), X[err_idx[i]])
    print(confusion_matrix(y, y_pred))


def evaluate():
    # normal: 0
    # spiderweb: 1
    X = []
    y = []
    X, y = readImageAndLabel("正常", 0, X, y)
    X, y = readImageAndLabel("蛛網", 1, X, y)
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
    printMispred(X, y, model)


def loadModel(model_filename):
    # 正常: 0
    # 蛛網: 1
    try:
        _ = open(model_filename)
        print("Load model from %s file." % model_filename)
    except:
        print("%s not found: start training." % model_filename)
        X, y = readImageAndLabel("正常", 0, X, y)
        X, y = readImageAndLabel("蛛網", 1, X, y)
        adj_X, adj_y = adjustData(X, y)
        model = SimpleCNN(200, 200)
        model.fit(X, y, batch_size=10, epochs=50)
        model.save(model_filename)
    model = load_model(model_filename)
    return model


def main():
    """
    Testing data:
        put testing data under /test/ directory
    """
    # load model
    model = loadModel("model.h5")
    test_data = readImage("test")
    test_data, _ = adjustData(test_data, None)
    y_pred = model.predict_classes(test_data)
    y_pred = ["正常" if not i else "蛛網" for i in y_pred]
    print(y_pred)


if __name__ == "__main__":
    main()
