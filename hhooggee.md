# 

<!-- 
{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "%matplotlib inline\n",
    "import matplotlib\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "import pandas as pd\n",
    "from sklearn.metrics import auc, confusion_matrix, roc_auc_score\n",
    "from sklearn.model_selection import train_test_split, GridSearchCV\n",
    "from sklearn.tree import DecisionTreeClassifier\n",
    "from sklearn.ensemble import RandomForestClassifier"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 顧客データCSVを読み込み前処理します"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 質問：　カテゴリデータの数値化は文字列ハッシュを使用してしまいましたが、予測結果に影響はでますでしょうか？"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "4 null data drop\n"
     ]
    }
   ],
   "source": [
    "dataset = pd.read_csv(\"customer.csv\")\n",
    "if dataset.isnull().sum().sum():\n",
    "    dataset = dataset.dropna()\n",
    "    print(\"null data drop\")\n",
    "# カテゴリデータを無理やり数値化します\n",
    "_ctg = [\"education\", \"marital_status\", \"dt_customer\"]\n",
    "dataset[_ctg] = dataset[_ctg].applymap(hash)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 質問１：STEP2でさらに７：３に分ける理由がよくわかってません"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 質問２：閾値を上げたり下げたり調整するのは何を見て調整すればよいのかわかってません。"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "## データ分割\n",
    "y = dataset[\"expected\"] #ターゲット変数\n",
    "X = dataset[[\"year_birth\",\"education\",\"marital_status\",\"income\",\"kidhome\",\"teenhome\",\"dt_customer\",\"recency\"]] # 特徴量変数\n",
    "###  step1：学習データを70%(X_train_val, y_train_val)、テストデータを30%(X_test, y_test)に分割します<br>\n",
    "X_train_val, X_test, y_train_val, y_test = train_test_split(\n",
    "    X, y, test_size=0.3, random_state=1234)\n",
    "###  step2：分割した学習データを、さらに70%(X_train, y_train)と30%(X_val, y_val)に分割します\n",
    "X_train, X_val, y_train, y_val = train_test_split(\n",
    "    X_train_val, y_train_val, test_size=0.3, random_state=1234)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_conf_matrix(threshold, y_proba, y_test):\n",
    "    y_pred = (y_proba[:, 1] > threshold).astype(int)\n",
    "    labels = [0, 1]\n",
    "    confusion_m = confusion_matrix(y_test, y_pred, labels=labels)\n",
    "    confusionm_df = pd.DataFrame(confusion_m, columns=labels, index=labels)\n",
    "    confusionm_df.rename(columns={0: \"predicted_0\", 1: \"predicted_1\"}, index={0: \"actual_0\", 1: \"actual_1\"}, inplace=True)\n",
    "    return confusionm_df\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "metadata": {},
   "outputs": [],
   "source": [
    "def tunning(rf, msg = \"\", X_dat = X_val, y_dat = y_val, threshold = 0.5):\n",
    "    rf.fit(X_train, y_train)\n",
    "    y_proba = rf.predict_proba(X_dat)\n",
    "    X_dat_copy = X_dat.copy()\n",
    "    X_dat_copy[\"予測プレミアム会員率\"] = y_proba[:, 1]\n",
    "    X_dat_copy[\"実際プレミアム会員\"] = y_dat\n",
    "    print(\"閾値\", threshold, \"の場合\")\n",
    "    print(get_conf_matrix(threshold, y_proba, y_dat))\n",
    "    print(msg, rf.__class__.__name__ + 'の評価')\n",
    "    print('AUC=', round(roc_auc_score(y_dat, y_proba[:, 1]), 3))\n",
    "    return X_dat_copy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "閾値 0.5 の場合\n",
      "          predicted_0  predicted_1\n",
      "actual_0          106            3\n",
      "actual_1            7            1\n",
      " DecisionTreeClassifierの評価\n",
      "AUC= 0.717\n"
     ]
    }
   ],
   "source": [
    "# 決定木モデルのライブラリ\n",
    "dt = DecisionTreeClassifier(random_state=1234, max_depth=3)\n",
    "ret = tunning(dt)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "閾値 0.5 の場合\n",
      "          predicted_0  predicted_1\n",
      "actual_0           98           11\n",
      "actual_1            7            1\n",
      " DecisionTreeClassifierの評価\n",
      "AUC= 0.512\n"
     ]
    }
   ],
   "source": [
    "# ランダムフォレストのライブラリ\n",
    "rf = DecisionTreeClassifier(random_state=1234)\n",
    "ret = tunning(rf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 3 folds for each of 25 candidates, totalling 75 fits\n",
      "閾値 0.5 の場合\n",
      "          predicted_0  predicted_1\n",
      "actual_0          108            1\n",
      "actual_1            7            1\n",
      "グリッドサーチでチューニング後 RandomForestClassifierの評価\n",
      "AUC= 0.767\n"
     ]
    }
   ],
   "source": [
    "# クロスバリデーションとグリッドサーチ のライブラリ\n",
    "rf = RandomForestClassifier(random_state=1234)\n",
    "### グリッドサーチでチューニングするパラメーターを用意します。\n",
    "### 最も平均精度が良いハイパーパラメータの組み合わせを確認しよう\n",
    "params = {'n_estimators': [10,15,20,25,30], 'max_depth': [10,15,20,25,30]}\n",
    "gscv = GridSearchCV(rf, param_grid=params, verbose=1, cv=3, scoring='roc_auc', n_jobs=-1)\n",
    "gscv.fit(X_train, y_train)\n",
    "best_parm = gscv.best_params_\n",
    "rf = RandomForestClassifier(random_state=1234, **best_parm)\n",
    "ret = tunning(rf, 'グリッドサーチでチューニング後')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# テストデータ使ってテスト！！"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "閾値 0.5 の場合\n",
      "          predicted_0  predicted_1\n",
      "actual_0          146            5\n",
      "actual_1           15            0\n",
      "テストデータへ適用して精度を確認 RandomForestClassifierの評価\n",
      "AUC= 0.599\n"
     ]
    }
   ],
   "source": [
    "rf = RandomForestClassifier(random_state=1234, **best_parm)\n",
    "ret = tunning(rf, 'テストデータへ適用して精度を確認', X_test, y_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 質問３：AUCがチューニングするたびに下がってしまっており意味が解りません。"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# sandbox cell"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>year_birth</th>\n",
       "      <th>education</th>\n",
       "      <th>marital_status</th>\n",
       "      <th>income</th>\n",
       "      <th>kidhome</th>\n",
       "      <th>teenhome</th>\n",
       "      <th>dt_customer</th>\n",
       "      <th>recency</th>\n",
       "      <th>予測プレミアム会員率</th>\n",
       "      <th>実際プレミアム会員</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>173</th>\n",
       "      <td>1967</td>\n",
       "      <td>-8059266157957462146</td>\n",
       "      <td>-7781548201323853673</td>\n",
       "      <td>79146.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>4378221832212863244</td>\n",
       "      <td>33</td>\n",
       "      <td>0.066667</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>461</th>\n",
       "      <td>1968</td>\n",
       "      <td>8290043827418450826</td>\n",
       "      <td>-6798223195665092031</td>\n",
       "      <td>50616.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1277156325954809714</td>\n",
       "      <td>56</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>154</th>\n",
       "      <td>1989</td>\n",
       "      <td>8290043827418450826</td>\n",
       "      <td>756988289848499807</td>\n",
       "      <td>77845.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>794465874117983338</td>\n",
       "      <td>40</td>\n",
       "      <td>0.033333</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>479</th>\n",
       "      <td>1984</td>\n",
       "      <td>8290043827418450826</td>\n",
       "      <td>-7036861153362654193</td>\n",
       "      <td>73356.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>-6553989075200428887</td>\n",
       "      <td>56</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>333</th>\n",
       "      <td>1948</td>\n",
       "      <td>-8059266157957462146</td>\n",
       "      <td>-7036861153362654193</td>\n",
       "      <td>92344.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>-2357772931227343321</td>\n",
       "      <td>9</td>\n",
       "      <td>0.600000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>170</th>\n",
       "      <td>1954</td>\n",
       "      <td>7632703312145897694</td>\n",
       "      <td>-7036861153362654193</td>\n",
       "      <td>63564.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>6482877679390168135</td>\n",
       "      <td>0</td>\n",
       "      <td>0.350000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>348</th>\n",
       "      <td>1969</td>\n",
       "      <td>8290043827418450826</td>\n",
       "      <td>756988289848499807</td>\n",
       "      <td>54132.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3636451025412482215</td>\n",
       "      <td>81</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>125</th>\n",
       "      <td>1964</td>\n",
       "      <td>-8059266157957462146</td>\n",
       "      <td>6416129981247406657</td>\n",
       "      <td>85620.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>-4127587453892149677</td>\n",
       "      <td>68</td>\n",
       "      <td>0.233333</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>437</th>\n",
       "      <td>1963</td>\n",
       "      <td>8290043827418450826</td>\n",
       "      <td>-7781548201323853673</td>\n",
       "      <td>48918.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>8595500041734391117</td>\n",
       "      <td>21</td>\n",
       "      <td>0.200000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>513</th>\n",
       "      <td>1976</td>\n",
       "      <td>8945785594254517079</td>\n",
       "      <td>-7036861153362654193</td>\n",
       "      <td>53204.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>-626619673379255897</td>\n",
       "      <td>40</td>\n",
       "      <td>0.000000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>166 rows × 10 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     year_birth            education       marital_status   income  kidhome  \\\n",
       "173        1967 -8059266157957462146 -7781548201323853673  79146.0        1   \n",
       "461        1968  8290043827418450826 -6798223195665092031  50616.0        0   \n",
       "154        1989  8290043827418450826   756988289848499807  77845.0        0   \n",
       "479        1984  8290043827418450826 -7036861153362654193  73356.0        0   \n",
       "333        1948 -8059266157957462146 -7036861153362654193  92344.0        0   \n",
       "..          ...                  ...                  ...      ...      ...   \n",
       "170        1954  7632703312145897694 -7036861153362654193  63564.0        0   \n",
       "348        1969  8290043827418450826   756988289848499807  54132.0        0   \n",
       "125        1964 -8059266157957462146  6416129981247406657  85620.0        0   \n",
       "437        1963  8290043827418450826 -7781548201323853673  48918.0        1   \n",
       "513        1976  8945785594254517079 -7036861153362654193  53204.0        1   \n",
       "\n",
       "     teenhome          dt_customer  recency  予測プレミアム会員率  実際プレミアム会員  \n",
       "173         1  4378221832212863244       33    0.066667          0  \n",
       "461         1  1277156325954809714       56    0.000000          0  \n",
       "154         0   794465874117983338       40    0.033333          0  \n",
       "479         0 -6553989075200428887       56    0.000000          1  \n",
       "333         0 -2357772931227343321        9    0.600000          0  \n",
       "..        ...                  ...      ...         ...        ...  \n",
       "170         0  6482877679390168135        0    0.350000          1  \n",
       "348         1  3636451025412482215       81    0.000000          0  \n",
       "125         0 -4127587453892149677       68    0.233333          1  \n",
       "437         1  8595500041734391117       21    0.200000          0  \n",
       "513         1  -626619673379255897       40    0.000000          0  \n",
       "\n",
       "[166 rows x 10 columns]"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "ret"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3.9.13 64-bit ('3.9.13')",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.13"
  },
  "orig_nbformat": 4,
  "vscode": {
   "interpreter": {
    "hash": "5fcd78efc9f118670fc3ce9d32556a7c4f66cdfbdb6df3b437a4f15b967bdcbf"
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
 -->
 
 
