# Common packages for answering all questions
import csv
import numpy as np
from numpy import pi
import pandas as pd
import sklearn
from numpy import arange, round, meshgrid, resize
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow.compat.v1 as tf # needed to get around placeholder bug in tf V2
tf.disable_v2_behavior()
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn import svm, metrics , datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Read in data for cleveland heart disease database
# filename = "C:/Users/antih/OneDrive - The University Of Newcastle/UoN files/COMP6380/Assignment/ANNs and SVMs/processed.cleveland.csv"
"""
def read_dataset(filename= "C:/Users/antih/OneDrive - The University Of Newcastle/UoN files/COMP6380/Assignment/ANNs and SVMs/processed.cleveland.csv"):
    x = []
    y = []
    with open(filename) as file:
        reader = csv.reader(file)
        #class_names = next(reader)[2:]
        for row in reader:
            x.append(list(map(float,row[:-1]))) # x is all the predictors
            y.append(int(row[-1])) # y is the classification

    return x, y

y = read_dataset()

"""

heart_disease = pd.read_csv(
    "C:/Users/antih/OneDrive - The University Of Newcastle/UoN files/COMP6380/Assignment/ANNs and SVMs/processed.cleveland.csv",
    names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca',
           'thal', 'classifier'],
    na_values={'?'})
# there are 6 NA values in this dataset
# just delete them for ease sake
heart_disease = heart_disease.dropna()

# split into x and y
y = heart_disease.pop('classifier').values
# Scikit-Learn needs 2d data so need to turn pandas df into a 2d array for modelling
x = heart_disease.iloc[:, 1:12].copy()

"""
code chunks which worked at converting data but fail when using with Scikit learn

#x = np.array(heart_disease.iloc[:,[13]])
#y = np.array(heart_disease.iloc[:,1:12])

#convert to lists to use with svm
#x = x.flatten()
#x = x.tolist()
#y = y.tolist()
#y = y.flatten()

"""

#SVM for data set 1

# split into test and train sets
x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=0)

classifier = svm.SVC(kernel='rbf', C=0.01)
predictions = classifier.fit(x_train, y_train).predict(x_test)

# build confusion matrix
confusion_matrix = np.zeros((len(y), len(y)))

for true_label, predicted_label in zip(y_test, predictions):
    confusion_matrix[true_label][predicted_label] +=1

# Plot non-normalized confusion matrix
titles_options = [("Confusion matrix, without normalization", None),
                  ("Normalized confusion matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(classifier, x_test, y_test,
                                 #display_labels=class_names,
                                 cmap=plt.cm.Blues,
                                 normalize=normalize)
    disp.ax_.set_title(title)

    print(title)
    print(disp.confusion_matrix)

plt.show()

accuracy =svm.metrics.accuracy_score(y_test, y_test, normalize=True, sample_weight=None)

print(accuracy)

sklearn.__path__