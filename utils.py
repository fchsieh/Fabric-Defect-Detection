import atexit
from datetime import timedelta
from distutils.util import strtobool
from time import localtime, strftime, time

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from sklearn.metrics import confusion_matrix, f1_score


class TimeEval:
    def __init__(self):
        self.start = time()

    def secondsToStr(self):
        return strftime("%Y-%m-%d %H:%M:%S", localtime())

    def log(self, s, display_elapsed=False):
        print("=" * 40)
        print(self.secondsToStr(), "-", s)
        if display_elapsed is True:
            elapsed = time() - self.start
            print("Elapsed time:", str(timedelta(seconds=elapsed)))
        print("=" * 40)


def user_yes_no_query(question):
    print("%s [y/n]" % question)
    while True:
        try:
            return strtobool(input().lower())
        except ValueError:
            print("Please respond with 'y' or 'n'.\n")


class AnalyzeResult:
    def __init__(self, truth, pred, classes_name, file_name):
        self.ground_truth = truth
        self.pred = pred
        self.classes_name = classes_name
        self.file_name = file_name

    def Analyzing_Test_Predict(self, verbose=True):
        result_str = []
        print("\n")
        err_idx = []
        for idx, pred in enumerate(self.pred):
            if pred != self.ground_truth[idx]:
                err_idx.append(idx)
        i = 0
        for idx, pred in enumerate(self.pred):
            if pred != self.ground_truth[idx]:
                testfn = "  Test %s" % self.file_name[err_idx[i]]
                real_pred = "Real: %s  |  Predict: %s" % (
                    self.classes_name[self.ground_truth[idx]],
                    self.classes_name[pred],
                )
                if verbose is True:
                    print(testfn.ljust(75), real_pred.rjust(24))
                result_str.append(testfn.ljust(75) + real_pred.rjust(24))
                i += 1
        with open("result.txt", "w") as file:
            for line in result_str:
                file.write(line + "\n")

        self.PrintConfusionMatrix()

    def PrintConfusionMatrix(self):
        pd.options.display.float_format = "${:20,0f}".format
        df_cm = pd.DataFrame(
            confusion_matrix(self.ground_truth, self.pred),
            index=self.classes_name,
            columns=self.classes_name,
        )
        plt.figure(figsize=(10, 7))
        sn.heatmap(df_cm, annot=True)
        print("\nShow and Save Confusion_Matrix.jpg")
        plt.savefig("confusion_matrix.jpg")
        print(confusion_matrix(self.ground_truth, self.pred))
        print("F1_score:  \n", f1_score(self.ground_truth, self.pred, average=None))


PHASE1_STR_DETECT_FILE = "\n\n=================================================================\
                              \n                           Phase I\
                              \n-----------------------------------------------------------------\
                              \nDetecting image files...\
                              \nUpdating Classes by folder name..."

PHASE2_STR_LOAD_IMAGE = "\n\n=================================================================\
                              \n                           Phase II\
                              \n-----------------------------------------------------------------\
                              \nLoading image files..."

PHASE3_STR_SPILT_DATA = "\n\n=================================================================\
                              \n                           Phase III\
                              \n-----------------------------------------------------------------\
                              \nSeperating Data to Training and Testing..."

PHASE4_STR_ADJUST_DATA = "\n\n=================================================================\
                              \n                           Phase IV\
                              \n-----------------------------------------------------------------\
                              \nNormalizing Image...\
                              \nCatagoring Label..."

# \nHistogram equalizing..."
PHASE5_STR_LOAD_MODEL = "\n\n=================================================================\
                              \n                           Phase V\
                              \n-----------------------------------------------------------------\
                              \nSetting Model..."

PHASE5_STR_TRAIN_MODEL = "\nTraining Model..."

PHASE6_STR_RUN_PRED = "\n\n=================================================================\
                              \n                           Phase VI\
                              \n-----------------------------------------------------------------\
                              \nRunning Testing Data..."

PHASE6_STR_ANALZYE_PRED = "\n\n=================================================================\
                              \n                           Phase VII\
                              \n-----------------------------------------------------------------\
                              \nAnalzying Testing Result..."
