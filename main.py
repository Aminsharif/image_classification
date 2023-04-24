import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# input_dir = os.getcwd()+'\\images'

project_dir = os.getcwd()
folder = 'images'
input_dir = os.path.join(project_dir, folder)
print(project_dir)
categories = ['empty', 'not_empty']

data = []
labels = []

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15,15))
        data.append(img)
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)