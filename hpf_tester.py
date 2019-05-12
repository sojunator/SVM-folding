from sklearn import svm
from sklearn import datasets
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler,  Normalizer
from sklearn.datasets import make_blobs, load_breast_cancer

import matplotlib.pyplot as plt
import numpy as np
import math
import json
import warnings
import pdb
import os
from hpf_helpers import plot, read_data_from_folder, clean_data, plot_3d, test_dataset, normalize_data, extend_data_spherical, plot2d_from_columns


nr_of_folds = 5


#Exception when divide when zero
np.seterr(all='warn')
warnings.filterwarnings('error')


data_set = read_data_from_folder("datasets") # Load data

data_points, data_labels = data_set["bmi.csv"]
data_points = normalize_data(data_points)
test_dataset(data_points, data_labels, "bmi", nr_of_folds, True)

"""
# Liver dataset

data_points, data_labels = data_set["liver.csv"]

data_points = normalize_data(data_points)


test_dataset(data_points, data_labels, "liver", nr_of_folds)


#Breast cancer
data = load_breast_cancer()
data_points  = data.data
data_labels = data.target

data_points = normalize_data(data_points)


test_dataset(data_points, data_labels, "cancer", nr_of_folds)


#hepatitis
data_points, data_labels = data_set["hepatitis.csv"]

data_points = normalize_data(data_points)

test_dataset(data_points, data_labels, "hepatitis", nr_of_folds)
"""
