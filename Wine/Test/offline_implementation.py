from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter

import numpy as np


# fetch dataset 
wine_quality = fetch_ucirepo(id=186) 


#prepare data


#plot feature distributrions to get an idea how they should be normalize

""" columns = ["fixed_acidity",
           "volatile_acidity",
           "citric_acid",
           "residual_sugar",
           "chlorides",
           "free_sulfur_dioxide",
           "total_sulfur_dioxide",
           "density",
           "pH",
           "sulphates",
           "alcohol",
           ] """

columns = wine_quality.data.features.columns

def visualize_dataset(features, targets, columns):
    fig, axis = plt.subplots(3, 4)
    c = 0
    for i in range(3):
        for j in range(4):
            if c == 11:
                break
            axis[i,j].title.set_text(columns[c])
            features[columns[c]].hist(bins = 20, ax = axis[i,j])
            c += 1
    axis[2,3].title.set_text("quality")
    targets["quality"].hist(bins = 20, ax = axis[2,3])
    plt.show()


#x = wine_quality.data.features.values #returns a numpy array

#df_features = wine_quality.data.features
df_features = pd.concat([wine_quality.data.features, wine_quality.data.targets], axis = 1)

drop_col = ["fixed_acidity",
           "citric_acid",
           "total_sulfur_dioxide",
           "density",
           "pH",
           "sulphates",
           ]


columns = ["fixed_acidity",
           "volatile_acidity",
           "citric_acid",
           "residual_sugar",
           "chlorides",
           "free_sulfur_dioxide",
           "total_sulfur_dioxide",
           "density",
           "pH",
           "sulphates",
           "alcohol",
           ]

df_features.drop(columns = drop_col, inplace = True)
df_features.drop_duplicates(inplace = True)

X, y = df_features.iloc[:,:-1], df_features.iloc[:,-1:]
#y = wine_quality.data.targets
#sort qquality into three classes, poor, average and good
v = y.values.squeeze()
#v1 = np.select([v<6,v==6, v>6],[0,1,2])[:,None]
v1 = np.select([v>6, v<=6],[1,0])[:,None]
y = pd.DataFrame(v1, columns=["quality"])
#y = wine_quality.data.targets - 3

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, stratify=y, random_state=100)

df4 = X_train.copy()
df4["quality"] = y_train.values
#yf = y_train.copy()

""" for i in [i for i in df4.columns]:
    if df4[i].nunique()>=12:
        Q1 = df4[i].quantile(0.06)
        Q3 = df4[i].quantile(0.94)
        IQR = Q3 - Q1
        df4 = df4[df4[i] <= (Q3+(1.5*IQR))]
        df4 = df4[df4[i] >= (Q1-(1.5*IQR))]

X_train = df4.iloc[:,:-1] 
y_train = df4.iloc[:,-1:]  """

#balance classes for training
us = RandomUnderSampler(sampling_strategy={0:1105,1:1100,2:738})
#sm = RandomOverSampler(sampling_strategy={0:1205,1:1599,2:1000})
sm = SMOTE(sampling_strategy={0:2804,1:2804})
print(np.unique(y_train.values, return_counts=True))
#X_train, y_train = us.fit_resample(X_train, y_train)
X_train, y_train = sm.fit_resample(X_train, y_train)
print(np.unique(y_train.values, return_counts=True))
#X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=100)
""" visualize_dataset(X_train, y_train, columns)
visualize_dataset(X_val, y_val, columns)
visualize_dataset(X_test, y_test, columns) """



""" es = xgb.callback.EarlyStopping(
    rounds=20,
    min_delta=1e-10,
    save_best=True,
    maximize=False,
    data_name="validation_0",
    metric_name="mlogloss",
) """
""" clf = XGBClassifier()
param_dist = {
    "n_estimators":[5,20,100,500, 1000],
    "max_depth":[1,3,5,7,9, None],
    #"learning_rate":[0.01,0.1,1,10,100],
} """


clf = RandomForestClassifier()
param_dist = {
    "n_estimators":[5,20,100, 250, 500, 1000],
    "max_depth":[1,3,5,7,9, None],
}

from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
 
#GB = GradientBoostingClassifier()
clf.fit(X_train, y_train)

""" cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
RCV = RandomizedSearchCV(clf, param_dist, n_iter=50, scoring='precision', n_jobs=-1, cv=2) 
clf = RCV.fit(X_train.values, y_train.values.ravel()).best_estimator_
"""

# Save model into JSON format.
importance = {'f0': 0.0, 
       'f1': 0.0, 
       'f2': 0.0, 
       'f3': 0.0, 
       'f4': 0.0, 
       'f5': 0.0, 
       'f6': 0.0, 
       'f7': 0.0, 
       'f8': 0.0, 
       'f9': 0.0, 
       'f10': 0.0}


predicted_proba = clf.predict_proba(X_test)
predicted = (predicted_proba [:,1] >= 0.7).astype('int')
#print(accuracy_score(y_test, predicted))

""" tests = 1
for i in range(tests):
    clf = RCV.fit(X_train.values, y_train.values.ravel()).best_estimator_
    print(accuracy_score(y_test, clf.predict(X_test)))
    importances = clf.get_booster().get_score(importance_type= "gain")
    print(plot_importance)
    for key, value in importances.items():
        importance[key] = importance[key] + value

for key, value in importance.items():
        importance[key] = importance[key] / tests """
#print(importance)

#clf.save_model("clf.json")
#clf.load_model("clf.json")
f = "gain"
#v = clf.get_booster().get_score(importance_type= f)
#plot_importance(clf)
#plt.show()
predicted
accuracy = accuracy_score(y_test, predicted)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, predicted))
print(accuracy)
#print(v)
pass

""" rf = RandomForestClassifier()
rf.fit(X_train, y_train)
preds = rf.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(accuracy) """

#Alcohol
#Volatile Acidity
#free sulfur dioxide
#residual sugar
