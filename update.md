# this test
<!-- 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import auc, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from itertools import permutations

dataset = pd.read_csv("ds_train.csv")
if "id1" in dataset.columns:
    del dataset["id1"]

if dataset.isnull().sum().sum():
    dataset = dataset.dropna()
    print("null data drop")
dataset = pd.get_dummies(dataset)

## データ分割
y = dataset["expected"] #ターゲット変数
X = dataset[[x for x in dataset.columns if x not in ("index","id", "expected")]] # 特徴量変数はターゲット変数以外全部
###  step1：学習データを70%(X_train_val, y_train_val)、テストデータを30%(X_test, y_test)に分割します<br>
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1234)
###  step2：分割した学習データを、さらに70%(X_train, y_train)と30%(X_val, y_val)に分割します
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.3, random_state=1234)

def get_conf_matrix(threshold, y_proba, y_test):
    y_pred = (y_proba[:, 1] > threshold).astype(int)
    confusion_m = confusion_matrix(y_test, y_pred)
    return confusion_m

def wsha_kpi(cm):
    TN, FP, FN, TP = list(cm.flatten())
    return 10 * TP - FP

def model_score(rf, name = "", threshold = 0.5, X_dat = X_val, y_dat = y_val):
    rf.fit(X_train, y_train)
    y_proba = rf.predict_proba(X_dat)
    cm = get_conf_matrix(threshold, y_proba, y_dat)
    profit = wsha_kpi(cm)
    auc = round(roc_auc_score(y_dat, y_proba[:, 1]), 3)
    print(f'{name}_閾値{threshold}\n\tW社のKPI: {profit}千円\n\tAUC={auc}')
    return dict(kpi=profit*auc, profit=profit, auc=auc, threshold=threshold, conf_matrix=cm, name=f"{name}/{threshold}")

EVAL = []

for i in range(1, 10):
    threshold = i / 10

    # 決定木モデルのライブラリ default param
    dt = DecisionTreeClassifier(random_state=1234)
    EVAL.append(model_score(dt, "decisiontree", threshold))

    # tuned maxdepth
    for a in range(10):
        dt = DecisionTreeClassifier(random_state=1234, max_depth=a)
        EVAL.append(model_score(dt, "decisiontree", threshold))

    # ランダムフォレストのライブラリ default param
    rf = RandomForestClassifier(random_state=1234)
    EVAL.append(model_score(rf, "randomforest", threshold))

    # クロスバリデーションとグリッドサーチ のライブラリ
    ## 最も平均精度が良いハイパーパラメータの組み合わせを確認しよう
    # params = {'n_estimators': [10,15,20,25,30], 'max_depth': [10,15,20,25,30]}
    # gscv = GridSearchCV(rf, param_grid=params, verbose=1, cv=3, scoring='roc_auc', n_jobs=6)
    # gscv.fit(X_train, y_train)
    # best_parm = gscv.best_params_
    # grf = RandomForestClassifier(random_state=1234, **best_parm)
    # model_score(grf, 'tunedrandomforest', threshold)

    for a, b in permutations(range(10, 30, 5), 2):
        rdic = dict(n_estimators=a, max_depth=b)
        rf = RandomForestClassifier(random_state=1234, **rdic)
        rdic.update(model_score(rf, "randomforest", threshold))
        EVAL.append(rdic)


EVAL.sort(key=lambda x: x["kpi"], reverse=True)
print(EVAL)
# rf = RandomForestClassifier(random_state=1234, **best_parm)
# tunning(rf, 'evaluate_testdata', X_test, y_test)

 -->
