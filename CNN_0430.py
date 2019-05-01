#!/usr/bin/env python
# coding: utf-8

import os
from glob import glob

try:
    import plaidml.keras

    plaidml.keras.install_backend()
except:
    import keras

import numpy as np
from keras.applications.resnet50 import ResNet50
from keras.layers import Conv2D, Flatten, MaxPooling2D
from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.utils import Sequence, to_categorical
from skimage.io import imread
from sklearn.model_selection import train_test_split

import utils


def adjust_img(x):
    X = np.asarray(x)
    X = X.astype("float32")
    X = X / 255
    return X


class My_Generator(Sequence):
    def __init__(self, image_filenames, labels, batch_size, classes):
        self.image_filenames, self.labels = image_filenames, labels
        self.batch_size = batch_size
        self.classes = classes

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        batch_x = self.image_filenames[
            idx * self.batch_size : (idx + 1) * self.batch_size
        ]
        batch_y = self.labels[idx * self.batch_size : (idx + 1) * self.batch_size]
        return (
            adjust_img(np.array([imread(file_name) for file_name in batch_x])),
            to_categorical(np.array(batch_y), num_classes=self.classes + 1),
        )


class DataSet:
    def __init__(self):
        self.classes = 0
        self.img_label = []
        self.img_filename = []
        self.data_size = 0
        self.classes_name = []
        # For evaluate_model model performance
        self.img_train_filename = []
        self.img_train_label = []
        self.img_test_filename = []
        self.img_test_label = []
        self.train_idx = []
        self.test_idx = []

    def read_img(self, datadir_path, testing_dataset=False):
        if not testing_dataset:
            dataset_dir = os.path.abspath(os.path.join(datadir_path))
            sub_dir = next(os.walk(dataset_dir))[1]
            for dirs in sub_dir:
                sub_dir_path = os.path.join(dataset_dir, dirs)
                img_files = glob(os.path.join(sub_dir_path, "*.bmp"))
                print(
                    "Class #",
                    self.classes,
                    "%s:" % dirs,
                    len(img_files),
                    "Image Found.",
                )
                for files in img_files:
                    self.img_filename.append(files)
                    self.img_label.append(self.classes)
                self.classes += 1
                self.classes_name.append(dirs)
            self.data_size = len(self.img_filename)
            print("Total %d images found." % self.data_size)
        else:
            # read testing images
            test_dir_path = os.path.abspath(os.path.join(datadir_path))
            for root, dirs, files in os.walk(test_dir_path):
                for file in files:
                    if file.endswith(".bmp"):
                        self.img_filename.append(os.path.join(root, file))

            self.data_size = len(self.img_filename)

            if self.data_size == 0:
                raise FileNotFoundError(
                    "Test images not found! Place images under /test/ directory."
                )
            else:
                print("Total %d images found." % self.data_size)

    def split_train_test(self, test_size=0.25):
        indices = np.arange(len(self.img_filename))
        self.img_train_filename, self.img_test_filename, self.img_train_label, self.img_test_label, self.train_idx, self.test_idx = train_test_split(
            self.img_filename,
            self.img_label,
            indices,
            random_state=42,
            test_size=test_size,
        )
        print("Training data: %d images." % (len(self.img_train_filename)))
        print("Testing data: %d images." % (len(self.img_test_filename)))


def SimpleCNN(sizeX, sizeY, classes):
    """
        Simple CNN Structure
    """
    model = Sequential()
    model.add(Conv2D(50, (3, 3), activation="relu", input_shape=(sizeX, sizeY, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(50, (3, 3), activation="relu", input_shape=(sizeX, sizeY, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(100, (3, 3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(100, (3, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(100, activation="relu"))
    model.add(Dense(100, activation="relu"))
    model.add(Dense(classes + 1, activation="softmax"))
    model.compile(loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
    return model


def train_model(model_filename, data_filename, label, classes):
    print(utils.PHASE4_STR_ADJUST_DATA)
    # Build data generator
    generator = My_Generator(data_filename, label, 100, classes)
    # fit data
    """
    for trainig built-in ResNet50...
    model = ResNet50(
        include_top=True,
        weights=None,
        input_shape=(200, 200, 3),
        classes=dataset.classes + 1)
    model.compile(
        loss="binary_crossentropy", optimizer="Adam", metrics=["accuracy"])
        """
    print(utils.PHASE5_STR_TRAIN_MODEL)
    execution_time = utils.TimeEval()
    execution_time.log("Start training model")
    model = SimpleCNN(200, 200, classes)
    model.fit_generator(
        generator=generator,
        steps_per_epoch=np.ceil((len(data_filename)) / 100),
        epochs=50,
        verbose=1,
        validation_data=None,
        validation_steps=None,
        use_multiprocessing=False,
    )
    model.save(model_filename)
    execution_time.log("End of training model", display_elapsed=True)


def evaluate_model(model_filename):
    print(utils.PHASE1_STR_DETECT_FILE)
    dataset = DataSet()
    print(utils.PHASE2_STR_LOAD_IMAGE)
    dataset.read_img("data")
    print(utils.PHASE3_STR_SPILT_DATA)
    dataset.split_train_test(test_size=0.25)

    # Check if parameter file already exists:
    if os.path.isfile(model_filename):
        print('Model "%s" found.' % model_filename)
        retrain_check = utils.user_yes_no_query("Retrain Model?")
    else:
        print('Model "%s" not found: start training.' % model_filename)
        print('Training Model "%s"......' % model_filename)
        retrain_check = 1

    if retrain_check == 1:
        train_model(
            "model.h5",
            dataset.img_train_filename,
            dataset.img_train_label,
            dataset.classes,
        )

    model = load_model(model_filename)
    print(utils.PHASE6_STR_RUN_PRED)
    img_test_data = adjust_img(np.array([imread(i) for i in dataset.img_test_filename]))
    test_pred = model.predict_classes(img_test_data)
    analyze = utils.AnalyzeResult(
        dataset.img_test_label,
        test_pred,
        dataset.classes_name,
        dataset.img_test_filename,
    )
    print(utils.PHASE6_STR_ANALZYE_PRED)
    analyze.Analyzing_Test_Predict(verbose=False)

    # predict all dataset
    all_datas = DataSet()
    all_datas.read_img("data")
    img_test_data = adjust_img(np.array([imread(i) for i in all_datas.img_filename]))
    test_pred = model.predict_classes(img_test_data)
    analyze = utils.AnalyzeResult(
        all_datas.img_label, test_pred, all_datas.classes_name, all_datas.img_filename
    )
    analyze.Analyzing_Test_Predict(verbose=False)


def predict(model_filename):
    if os.path.isfile(model_filename):
        print('Model "%s" found.' % model_filename)
        retrain_check = utils.user_yes_no_query("Retrain Model?")
    else:
        print('Model "%s" not found: start training.' % model_filename)
        print('Training Model "%s"......' % model_filename)
        retrain_check = 1

    if retrain_check == 1:
        dataset = DataSet()
        dataset.read_img("data")
        train_model(
            "model.h5", dataset.img_filename, dataset.img_label, dataset.classes
        )

    model = load_model(model_filename)
    test_dataset = DataSet()
    try:
        if not os.path.isdir("test"):
            os.mkdir("test")
            raise NotADirectoryError(
                "Test directory not found! Place images under /test/ directory."
            )
        test_dataset.read_img("test", testing_dataset=True)
        img_test_data = adjust_img(
            np.array([imread(i) for i in test_dataset.img_filename])
        )
        test_pred = model.predict_classes(img_test_data)
        for fn, pred in zip(test_dataset.img_filename, test_pred):
            print(fn, ":", pred)
    except OSError as error:
        print(error)


def main():
    evaluate_model("model.h5")
    # predict("model.h5")


if __name__ == "__main__":
    main()
