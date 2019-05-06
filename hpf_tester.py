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
from hpf_helpers import plot, read_data_from_folder, clean_data, plot_3d, test_dataset





#Exception when divide when zero
np.seterr(all='warn')
warnings.filterwarnings('error')

data_set = read_data_from_folder("datasets") # Load data

data_points, data_labels = data_set["liver.csv"]

transformer = Normalizer().fit(data_points) # fit does nothing.
data_points_new = transformer.transform(data_points)




test_dataset(data_points_new, data_labels, "liver")



data = load_breast_cancer()
data_points  = data.data
data_labels = data.target

transformer = Normalizer().fit(data_points) # fit does nothing.
data_points_new = transformer.transform(data_points)


test_dataset(data_points_new, data_labels, "cancer")


data_points, data_labels = data_set["bmi.csv"]

transformer = Normalizer().fit(data_points) # fit does nothing.
data_points_new = transformer.transform(data_points)




#test_dataset(data_points_new, data_labels, "bmi")
