import json
import pickle
import pandas as pd
import random
import cv2
from scipy import misc
from PIL import Image
#from skimage import exposure
from sklearn import svm
from PIL import Image
from sklearn.decomposition import PCA

import scipy
from math import sqrt,pi
from numpy import exp
from matplotlib import pyplot as plt
import numpy as np
import glob
import matplotlib.pyplot as pltss
import cv2
from matplotlib import cm
import pandas as pd
from math import pi, sqrt
import os
#import pywt
from sklearn import svm
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score, classification_report, recall_score
from sklearn.metrics import cohen_kappa_score, accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso,LassoCV,LassoLarsCV,LogisticRegression
os.chdir('./Downloads/fintech_DR')
# load test set and training set

with open('test_set.pkl','rb') as f:
    test_set = pickle.load(f)
with open('training_set.pkl','rb') as f:
    training_set = pickle.load(f)   

labels = pd.concat([training_set,test_set],axis=0).reset_index(drop = True)

y = labels.label.tolist()
# create a newlabel: "1" indicates 2-4
labels['newlabel'] = labels['label'].apply(lambda x: 1 if x>=2 else 0)
y_new = labels.newlabel.tolist()

#list of image name and ignore hidden files
files = [x for x in os.listdir('500_enhanced_augmentated') if not x.startswith('.')] 


x=[]
# PCA
'''
for i in labels.filename:
    img = cv2.imread('500_enhanced_augmentated/' + i)
    
    x.append(np.array(img).flatten())



x=pd.DataFrame(np.matrix(x))
pca = PCA(n_components=500, whiten=True).fit(x)
pca_x=pca.transform(x)
'''

# k means
for file in labels.filename:
    img = cv2.imread('500_enhanced_augmentated/' + file)
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    k=cv2.KMEANS_PP_CENTERS
    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 10
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,k)
    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    x.append(np.array(res2).flatten())

# Training Test Spilt
x_train, x_test = x[:1200], x[1200:]
y_train, y_test = y[:1200], y[1200:]
y_train_bi, y_test_bi = y_new[:1200], y_new[1200:]
# SVM
clf = svm.SVC(C=0.01, kernel = 'rbf',decision_function_shape='ovr',gamma = 'auto')
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("test set accuracy: ",accuracy_score(y_test, y_pred))
print("quadratic weighted kappa: ", cohen_kappa_score(y_test, y_pred,weights='quadratic'))
print(confusion_matrix(y_test, y_pred))

# Logistic Regression + Lasso
log = LogisticRegression(penalty='l1', solver='liblinear')
log.fit(x_train, y_train)
y_lasso_pred = log.predict(x_test)
print("test set accuracy: ",accuracy_score(y_test, y_lasso_pred))
print(confusion_matrix(y_test, y_lasso_pred))
print("quadratic weighted kappa: ", cohen_kappa_score(y_test, y_lasso_pred,weights='quadratic'))

# Sensitivity/Recall = True Positive Rate
print("recall: ",recall_score(y_test, y_lasso_pred))
print(roc_curve(y_test,y_lasso_pred))

log = LogisticRegression(penalty='l1', solver='liblinear')
log.fit(x_train, y_train_bi)
y_lasso_pred_bi = log.predict(x_test)
print("test set accuracy: ",accuracy_score(y_test, y_lasso_pred_bi))
print(confusion_matrix(y_test_bi, y_lasso_pred_bi))
print("quadratic weighted kappa: ", cohen_kappa_score(y_test, y_lasso_pred_bi,weights='quadratic'))

# Sensitivity/Recall = True Positive Rate
print("recall: ",recall_score(y_test_bi, y_lasso_pred_bi))
print(roc_auc_score(y_test_bi,y_lasso_pred_bi))

# LDA
lda = LinearDiscriminantAnalysis()
lda.fit(x_train, y_train)
y_lda_pred = lda.predict(x_test)
print(confusion_matrix(y_test, y_lda_pred))
print(lda.score(x_test, y_test))
# training set accuracy
print(lda.score(x_train, y_train))

############## Quadratic Weighted Kappa
from functools import reduce
def confusion_matrix(rater_a, rater_b,
		 min_rating=None, max_rating=None):
    """
    Returns the confusion matrix between rater's ratings
    """
    assert(len(rater_a)==len(rater_b))
    if min_rating is None:
        min_rating = min(reduce(min, rater_a), reduce(min, rater_b))
    if max_rating is None:
        max_rating = max(reduce(max, rater_a), reduce(max, rater_b))
    num_ratings = max_rating - min_rating + 1
    conf_mat = [[0 for i in range(num_ratings)]
                for j in range(num_ratings)]
    for a,b in zip(rater_a,rater_b):
        conf_mat[a-min_rating][b-min_rating] += 1
    return conf_mat

def histogram(ratings, min_rating=None, max_rating=None):
    """
    Returns the counts of each type of rating that a rater made
    """
    if min_rating is None: min_rating = reduce(min, ratings)
    if max_rating is None: max_rating = reduce(max, ratings)
    num_ratings = max_rating - min_rating + 1
    hist_ratings = [0 for x in range(num_ratings)]
    for r in ratings:
	    hist_ratings[r-min_rating] += 1
    return hist_ratings

def quadratic_weighted_kappa(rater_a, rater_b,
                             min_rating = None, max_rating = None):
    """
    Calculates the quadratic weighted kappa
    scoreQuadraticWeightedKappa calculates the quadratic weighted kappa
    value, which is a measure of inter-rater agreement between two raters
    that provide discrete numeric ratings.  Potential values range from -1  
    (representing complete disagreement) to 1 (representing complete
    agreement).  A kappa value of 0 is expected if all agreement is due to
    chance.
    
    scoreQuadraticWeightedKappa(rater_a, rater_b), where rater_a and rater_b
    each correspond to a list of integer ratings.  These lists must have the
    same length.
    
    The ratings should be integers, and it is assumed that they contain
    the complete range of possible ratings.
   
    score_quadratic_weighted_kappa(X, min_rating, max_rating), where min_rating
    is the minimum possible rating, and max_rating is the maximum possible
    rating
    """
    assert(len(rater_a) == len(rater_b))
    if min_rating is None:
        min_rating = min(reduce(min, rater_a), reduce(min, rater_b))
    if max_rating is None:
        max_rating = max(reduce(max, rater_a), reduce(max, rater_b))
    conf_mat = confusion_matrix(rater_a, rater_b,
				     min_rating, max_rating)
    num_ratings = len(conf_mat)
    num_scored_items = float(len(rater_a))

    hist_rater_a = histogram(rater_a, min_rating, max_rating)
    hist_rater_b = histogram(rater_b, min_rating, max_rating)

    numerator = 0.0
    denominator = 0.0

    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i]*hist_rater_b[j]
			     / num_scored_items)
            d = pow(i-j,2.0) / pow(num_ratings-1, 2.0)
        numerator += d*conf_mat[i][j] / num_scored_items
        denominator += d*expected_count / num_scored_items

    return 1.0 - numerator / denominator

print("quadratic weighted kappa: ", quadratic_weighted_kappa(y_pred,y_test))
