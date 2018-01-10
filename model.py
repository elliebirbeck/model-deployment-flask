import scipy.io 
import numpy as np
from sklearn.utils import shuffle 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.externals import joblib

# load data file as dict object
train_data = scipy.io.loadmat('extra_32x32.mat') 

# extract the images (X) and labels (y) from the dict
X = train_data['X'] 
y = train_data['y'] 

# reshape our matrices into 1D vectors and shuffle (still maintaining the index pairings)
X = X.reshape(X.shape[0]*X.shape[1]*X.shape[2],X.shape[3]).T 
y = y.reshape(y.shape[0],) 
X, y = shuffle(X, y, random_state=42)

# split data into training and testing sets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05, random_state=42)

# define classifier and fit to training data
clf = RandomForestClassifier() 
clf.fit(X_train, y_train) 

# save model
joblib.dump(clf, 'model.pkl')
