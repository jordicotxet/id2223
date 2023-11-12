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

project = hopsworks.login()
fs = project.get_feature_store()

try:
    wine_fg = fs.get_feature_group(name="wine", version=1)
    query = wine_fg.select_all()
    feature_view = fs.get_or_create_feature_view(name="wine",
                                    version=1,
                                    description="Read from wine dataset",
                                    labels=["quality"],
                                    query=query)

    feature_view.delete()
except:
    print("Featureview deletion unsuccessful")

wine_fg = fs.get_feature_group(
    name="wine",
    version=1)
wine_fg.delete()

for key in [0, 1, 2]:
    fg = fs.get_feature_group(
    name="wine_subset_" + str(int(key)),
    version=1)
    fg.delete()





