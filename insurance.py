"""
DEĞİŞKENLER :

yaş: birincil yararlanıcının yaşı.

cinsiyet: sigorta yüklenicisi cinsiyeti, kadın, erkek.

bmı: Vücut kitle indeksi, vücudun, boyuna göre nispeten yüksek veya düşük ağırlıkların anlaşılmasını sağlar,
boy-kilo oranını kullanarak vücut ağırlığının objektif indeksi (kg / m ^ 2), ideal olarak 18,5 ila 24,9...

çocuklar: Sağlık sigortası kapsamındaki çocuk sayısı / Bakmakla yükümlü olunan çocuk sayısı.

sigara içen: Sigara içiyor mu ?

bölge: yararlanıcının ABD'deki yerleşim alanı, kuzeydoğu, güneydoğu, güneybatı, kuzeybatı.

ücretler: Sağlık sigortası tarafından faturalandırılan bireysel tıbbi masraflar.

"""

# Kullanılacak olan modüllerin import işlemi
import warnings
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error,mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

pd.set_option("Display.Max_columns", None)
pd.set_option("Display.width", 500)

# veri setinin okunması
insurance = pd.read_csv("insurance.csv")
df = insurance.copy()

# veri setine ve değişkenlere genel bakış
df.head()
df.shape
df.info()
df["sex"].value_counts()
df["region"].value_counts()
df["smoker"].value_counts()
df["children"].value_counts()

# eksik değer kontrolü
df.isnull().sum()

# değişkenlerin sayısal ve kategorik olarak ayrılması.
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]

# değişkenlerin birbirleri arasındaki kolerasyon ilişkisi.
sns.boxplot(df[num_cols])
plt.show(block=True)

# sayısal değerlerin istatiksel özetlerine bakış
df[num_cols].describe().T

# 2 sınıfa sahip değişkenlerin encode işlemi.
df["smoker"] = LabelEncoder().fit_transform(df["smoker"])
df["sex"] = LabelEncoder().fit_transform(df["sex"])

# 2 den fazla sınıfa sahip değişkenlerin encode işlemi.
ohe_cols = [col for col in df.columns if 10 >= df[col].nunique() > 2]
def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df,categorical_cols=ohe_cols,drop_first=True)

# encode sonrası tekrardan değişken atamaları ve scale işlemi için gerekli sütunların ataması
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
cols = [col for col in df.columns if "charges" not in col]
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]

# Kuracak olduğumuz modellerimizin performasını arttırmak amaçlı yeni değişkenler türetme işlemi.
df["age_sex_mean"] = df.groupby(["age", "sex"])["charges"].transform("mean")
df["smokers_age"] = df["smoker"] / df["age"]
df["smokers_char"] = df["smoker"] / df["charges"]
df["age_char"] = df["age"] / df["charges"]

# Üretilen yeni değişkenlerle birlikte tekrar değişken atamaları
num_cols = [col for col in df.columns if df[col].dtypes != "O"]
cols = [col for col in df.columns if "charges" not in col]
cat_cols = [col for col in df.columns if df[col].dtypes == "O"]


# MODELLEME


X = df.drop("charges", axis=1)
y = df[["charges"]]

# Standartlaştırmak için :
#X_scaled = StandardScaler().fit_transform(X)
#X = pd.DataFrame(X_scaled, columns=X.columns)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=20)


models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor()),
          ("LightGBM", LGBMRegressor(verbose=-1)),
          ("CatBoost", CatBoostRegressor(verbose=False))]

print("--- RMSE Skorları ---")
for name, regression in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regression, X_train, y_train, cv=5, scoring="neg_mean_squared_error")))
    print("-------------")
    print(f"{name}: {round(rmse, 3)} ")


knn_params = {"n_neighbors": [3]}

cart_params = {'max_depth': [15, 16, 18],
               "min_samples_split": [7]}

rf_params = {"max_depth": [13,14,15, None],
             "max_features": [10,12,14],
             "min_samples_split": [2],
             "n_estimators": [1000,1200,1400]}

gbm_params = {"learning_rate": [0.1],
              "max_depth": [12],
              "n_estimators": [100],
              "subsample": [0.4]}

xgboost_params = {"learning_rate": [0.02],
                  "max_depth": [3],
                  "n_estimators": [5000,6000,8000],
                  "colsample_bytree": [0.6]}

lgbm_params = {"learning_rate": [0.1],
               "n_estimators": [2000,2500,3000],
               "colsample_bytree": [0.7]}

catboost_params = {"iterations": [1200,1500,3000],
                   "learning_rate": [0.3],
                   "depth": [3]}

hiperparam = [
    ("KNN", GridSearchCV(KNeighborsRegressor(), knn_params, cv=5, n_jobs=-1, verbose=True, scoring='neg_mean_squared_error').fit(X_train, y_train)),
    ("CART", GridSearchCV(DecisionTreeRegressor(), cart_params, cv=5, n_jobs=-1, verbose=True, scoring='neg_mean_squared_error').fit(X_train, y_train)),
    ("RF", GridSearchCV(RandomForestRegressor(), rf_params, cv=5, n_jobs=-1, verbose=True, scoring='neg_mean_squared_error').fit(X_train, y_train)),
    ("GBM", GridSearchCV(GradientBoostingRegressor(), gbm_params, cv=5, n_jobs=-1, verbose=True, scoring='neg_mean_squared_error').fit(X_train, y_train)),
    ("XGB", GridSearchCV(XGBRegressor(), xgboost_params, cv=5, n_jobs=-1, verbose=True, scoring='neg_mean_squared_error').fit(X_train, y_train)),
    ("LGBM", GridSearchCV(LGBMRegressor(), lgbm_params, cv=5, n_jobs=-1, verbose=True, scoring='neg_mean_squared_error').fit(X_train, y_train)),
    ("CATBOOST", GridSearchCV(CatBoostRegressor(), catboost_params, cv=5, n_jobs=-1, verbose=True, scoring='neg_mean_squared_error').fit(X_train, y_train))
]

for name, hiper in hiperparam:
    best_params = hiper.best_params_
    best_score = np.sqrt(-hiper.best_score_)
    print(f"En iyi RMSE: {round(best_score, 3)}")


# CATBOOST

grid_cv = hiperparam[6][1]

catboost_final = CatBoostRegressor(**grid_cv.best_params_).fit(X_train, y_train)

y_pred = catboost_final.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))

mean_absolute_percentage_error(y_test, y_pred)

# LGBM
grid_cv = hiperparam[5][1]

lgbm_final = LGBMRegressor(**grid_cv.best_params_).fit(X_train, y_train)

y_pred = lgbm_final.predict(X_test)

np.sqrt(mean_squared_error(y_test, y_pred))

mean_absolute_percentage_error(y_test, y_pred)