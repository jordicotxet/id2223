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
from sklearn.inspection import permutation_importance
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
import hopsworks
from sklearn.metrics import confusion_matrix

import numpy as np



# fetch dataset 
wine_quality_white = pd.read_csv("Wine/winequality/winequality-white.csv", sep=";")
wine_quality_red = pd.read_csv("Wine/winequality/winequality-red.csv", sep=";")
for i in wine_quality_red.columns[wine_quality_red.isnull().any(axis=0)]:     #---Applying Only on variables with NaN values
    wine_quality_red[i].fillna(wine_quality_red[i].mean(),inplace=True)
#wine_quality_red.fillna(value=wine_quality_red.mean(axis = 0), inplace=True)
#wine_quality = np.concatenate([wine_quality_white.values, wine_quality_red.values], axis = 0)
#wine_quality = pd.DataFrame(wine_quality, columns=wine_quality_white.columns)
wine_quality = wine_quality_white
df_features = wine_quality

df_features = wine_quality

drop_col = ["fixed_acidity",
           "citric_acid",
           "total_sulfur_dioxide",
           "density",
           "pH",
           "sulphates",
           ]

df_features.drop(columns = drop_col, inplace = True)
df_features.drop_duplicates(inplace = True)
X, y = df_features.iloc[:,:-1], df_features.iloc[:,-1:]

""" columns = [
           "volatile_acidity",
           "residual_sugar",
           "chlorides",
           "free_sulfur_dioxide",
           "alcohol",
           "quality"
           ] """


v = y.values.squeeze()
v1 = v
#v1 = np.select([v<6,v==6, v>6],[0,1,2])[:,None]
v1 = np.select([v<6, v>6],[0,1])[:,None]
#v1 = np.select([v<5,(v>=5)&(v<=6), v>6],[0,1,2])[:,None]
#v1 = np.select([v<5,v==5, v==6,v==7,v>7],[0,1,2,3,4])[:,None]
y = pd.DataFrame(v1, columns=["quality"])
#X = X[['alcohol', 'sulphates']]

#scaler = preprocessing.MinMaxScaler()
scaler = preprocessing.StandardScaler()
#scaler = preprocessing.RobustScaler()
#x_scaled = scaler.fit_transform(X.values)
#X = pd.DataFrame(x_scaled, columns=X.columns)
#df_features = wine_quality.data.features


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

#df4 = X_train.copy()
#df4["quality"] = y_train.values
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
sm = RandomOverSampler()
#sm = SMOTE(sampling_strategy={0:2804,1:2804})
#sm = SMOTE()
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

""" param_dist = {
    "n_estimators":[250, 500, 1000, 2000],
    "max_depth":[1,3,5,7,9, 12, 14, 16, 20, None],
} """

""" param_dist = {
    "n_estimators":[250],
    "max_depth":[1],
} """

""" clf = RandomForestClassifier()
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=10)
RCV = GridSearchCV(clf, param_dist, n_jobs=-1, cv=cv) 
clf = RCV.fit(X, y.values.ravel()).best_estimator_
print(RCV.best_params_)
 """
clf = RandomForestClassifier(n_estimators=1000, max_depth=12)
#clf = MLPClassifier(hidden_layer_sizes=(30, 50))
""" cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3)
scores = cross_val_score(clf, X=X, y=y.values.ravel(), cv = cv, n_jobs=-1)
print("Accuracy mean:",scores.mean(), "std:",scores.std())
param_dist = {
    "n_estimators":[5,20,100, 250, 500, 1000],
    "max_depth":[1,3,5,7,9, None],
} """


project = hopsworks.login()
fs = project.get_feature_store()


wines = []
for i in range(3):
    name = "wine_subset_" + str(i)
    stat_fg = fs.get_feature_group(name=name, version=1)
    stats = stat_fg.get_statistics().content['columns']

    wines.append(generate_wine(stats, 1000))

 
generated = pd.concat(wines, ignore_index=True)
y_gen = generated['quality'].values
generated.drop(columns = ['quality'], inplace = True)
df = pd.DataFrame()
columns = [
           "volatile_acidity",
           "residual_sugar",
           "chlorides",
           "free_sulfur_dioxide",
           "alcohol",
           ]
for c in columns:
    df[c] = generated[c]
#GB = GradientBoostingClassifier()
generated = df
clf.fit(X_train, y_train)

predicted = clf.predict(X_test)
accuracy = accuracy_score(y_test, predicted)
#y_test = np.ones_like(y_test)
print(confusion_matrix(y_test, predicted))
print(accuracy)
predicted = clf.predict(generated)
accuracy = accuracy_score(y_gen, predicted)
#y_test = np.ones_like(y_test)
print(confusion_matrix(y_gen, predicted))
print(accuracy)


result = permutation_importance(clf, X_train, y_train, n_repeats=20,random_state=0, n_jobs=-1)
#print(result.importances_mean)
forest_importances = pd.Series(result.importances_mean, index=X_train.columns)

fig, ax = plt.subplots()
forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
ax.set_title("Feature importances using permutation on full model")
ax.set_ylabel("Mean accuracy decrease")
fig.tight_layout()
plt.show()

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


#predicted_proba = clf.predict_proba(X_test)
#predicted = (predicted_proba [:,1] >= 0.7).astype('int')
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
