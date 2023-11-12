from ucimlrepo import fetch_ucirepo 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import hopsworks

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier, plot_importance
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.over_sampling import SMOTE, RandomOverSampler
from collections import Counter
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold, RandomizedSearchCV

wine_quality = pd.read_csv("Wine/winequality/winequality-white.csv", sep=";")

#wine_quality = fetch_ucirepo(id=186) 

#df_features = pd.concat([wine_quality.data.features, wine_quality.data.targets], axis = 1)
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

columns = [
           "volatile_acidity",
           "residual_sugar",
           "chlorides",
           "free_sulfur_dioxide",
           "alcohol",
           "quality"
           ]


v = y.values.squeeze()
v1 = np.select([v<6,v==6, v>6],[0,1,2])[:,None]
y = pd.DataFrame(v1, columns=["quality"])

X_val = X.values
y_val = y.values
df_data = np.concatenate((X_val, y_val), axis = 1)

df = pd.DataFrame(df_data, columns=columns)

grouping = df.groupby('quality')
stats = {}

v2 = None 

project = hopsworks.login()
fs = project.get_feature_store()

wine_fg = fs.get_or_create_feature_group(
    name="wine",
    version=1,
    primary_key=columns, 
    description="Dataset containing properties of different white wines and their respective qualities")
#wine_fg.delete()
wine_fg.insert(df)

for key in grouping.groups.keys():
    #AA = "wine_stats_" + str(int(n))
    fg = fs.get_or_create_feature_group(
    name="wine_subset_" + str(int(key)),
    version=1,
    primary_key=columns, 
    description="Wine that belong to quality group: " +  str(key),
    parents=[wine_fg])
    v1 = grouping.get_group(key)
    #v2 = v1.loc['mean']
    #fg.delete()
    fg.insert(v1)
    #for i, ii in zip(v2.index, v2):
    #    pass
    #v2 = v1.values[1:3,:] if v2 is None else np.concatenate((v1.values[1:3,:], v2), axis = 0)

#v2 = np.concatenate((v2, np.arange(3).repeat(2)[:, None]))





