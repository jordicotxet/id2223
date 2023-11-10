import hopsworks
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV

from xgboost import XGBClassifier, plot_importance
import xgboost as xgb

import numpy as np


# You have to set the environment variable 'HOPSWORKS_API_KEY' for login to succeed
project = hopsworks.login()
fs = project.get_feature_store()

iris_fg = fs.get_feature_group(name="wine", version=1)
query = iris_fg.select_all()
feature_view = fs.get_or_create_feature_view(
    name="wine",
    version=1,
    labels = ["quality"], 
    query=query)

# You can read training data, randomly split into train/test sets of features (X) and labels (y)        
X_train, X_test, y_train, y_test = feature_view.train_test_split(0.5)
X = pd.concat([X_train, X_test], axis = 0)
clf = XGBClassifier()
param_dist = {
    "n_estimators":[5,20,100,500],
    "max_depth":[1,3,5,7,9],
    "learning_rate":[0.01,0.1,1,10,100]
}
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)

RCV = RandomizedSearchCV(clf, param_dist, n_iter=50, scoring='f1_weighted', n_jobs=-1, cv=2)
clf = RCV.fit(X_train.values, y_train.values.ravel()).best_estimator_
clf.save_model("clf.json")
clf.load_model("clf.json")
preds = clf.predict(X_test)
accuracy = accuracy_score(y_test, preds)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, preds))
print(accuracy)
