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
scaler = preprocessing.MinMaxScaler()
#scaler = preprocessing.RobustScaler()
x_scaled = scaler.fit_transform(wine_quality.data.features)
df_features = pd.DataFrame(x_scaled, columns=columns)
df_features = wine_quality.data.features

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
#df_features = wine_quality.data.features.transform(min_max_scaler.fit_transform)

#sns.histplot(data=wine_quality.data.features, x="fixed_acidity", bins= 12)
#Train
""" l = len(wine_quality.data.features)
trainsplit = 0.9
train_idx = int(trainsplit*l)

X = wine_quality.data.features
y = wine_quality.data.labels """
X = df_features
#y = wine_quality.data.targets
#sort qquality into three classes, poor, average and good
v = wine_quality.data.targets.values.squeeze()
v1 = np.select([v<6,v==6, v>6],[0,1,2])[:,None]
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
sm = RandomOverSampler(random_state=100)
print(np.unique(y_train.values, return_counts=True))
#X_train, y_train = sm.fit_resample(X_train, y_train)
print(np.unique(y_train.values, return_counts=True))
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, stratify=y_test, random_state=100)
""" visualize_dataset(X_train, y_train, columns)
visualize_dataset(X_val, y_val, columns)
visualize_dataset(X_test, y_test, columns) """


#x_train, x_test = wine_quality.data.features.iloc[:train_idx,:], wine_quality.data.features.iloc[train_idx:,:]
#y_train, y_test = wine_quality.data.features.iloc[:train_idx,:], wine_quality.data.features.iloc[train_idx:,:]
# data (as pandas dataframes) 
#X = wine_quality.data.features.iloc[:2,:]
#y = wine_quality.data.targets 

""" es = xgb.callback.EarlyStopping(
    rounds=20,
    min_delta=1e-10,
    save_best=True,
    maximize=False,
    data_name="validation_0",
    metric_name="mlogloss",
) """
clf = XGBClassifier()
# Fit the model, test sets are used for early stopping.
#clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])
param_dist = {
    "n_estimators":[5,20,100,500],
    "max_depth":[1,3,5,7,9],
    "learning_rate":[0.01,0.1,1,10,100]
}

from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV
 
#GB = GradientBoostingClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)

RCV = RandomizedSearchCV(clf, param_dist, n_iter=50, scoring='f1_weighted', n_jobs=-1, cv=cv)
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

tests = 1
for i in range(tests):
    clf = RCV.fit(X_train.values, y_train.values.ravel()).best_estimator_
    print(accuracy_score(y_test, clf.predict(X_test)))
    importances = clf.get_booster().get_score(importance_type= "gain")
    #print(plot_importance)
    for key, value in importances.items():
        importance[key] = importance[key] + value

for key, value in importance.items():
        importance[key] = importance[key] / tests
print(importance)

clf.save_model("clf.json")
clf.load_model("clf.json")
f = "gain"
v = clf.get_booster().get_score(importance_type= f)
plot_importance(clf)
#plt.show()
preds = clf.predict(X_test)
accuracy = accuracy_score(y_test, preds)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test, preds))
print(accuracy)
print(v)
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
