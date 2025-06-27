from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, multilabel_confusion_matrix
from sklearn.decomposition import PCA
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import csv
import numpy as np
from tqdm import tqdm 

file_path = 'Playground/colors_denoised_quantized.csv'

def train_in_chunks(file_path):
    chunk_size = 10 ** 6
    test_x = []
    test_y = []
    def get_total_rows(list_):
        sum = 0
        for batch in list_:
            sum += batch.shape[0]
        return sum
        
    for chunk in tqdm(pd.read_csv(file_path, chunksize=chunk_size)):
        X = chunk.iloc[:, :-1]
        y = chunk.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        test_x.append(np.array(X_test))
        test_y.append(np.array(y_test))
        clf = OneVsRestClassifier(LogisticRegression()).fit(X_train, y_train)
    test_x.pop()
    test_y.pop()
    big_test_x = np.array(test_x)
    big_test_y = np.array(test_y)
    big_test_x = big_test_x.reshape((get_total_rows(test_x), 5))
    big_test_y = big_test_y.reshape((get_total_rows(test_y), 1))
    y_pred = clf.predict(big_test_x)
    print(accuracy_score(big_test_y, y_pred))


train_in_chunks(file_path)