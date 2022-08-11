import time
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from matplotlib.ticker import AutoMinorLocator
import warnings

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
warnings.filterwarnings('ignore')

train = pd.read_csv('train.csv', parse_dates=['date'])
test = pd.read_csv('test.csv', parse_dates=['date'])

df = pd.concat([train, test], sort=False)

def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.quantile([0, 0.05, 0.50, 0.75, 0.95, 0.99, 1]).T)

check_df(train)

train.groupby(["store"])["item"].nunique()

train.groupby(["store", "item"]).agg({"sales": ["sum"]})

train.groupby(["store", "item"]).agg({"sales": ["sum", "mean", "median", "std"]})


train.head()

def create_date_features(df):
    df['month'] = df.date.dt.month
    df['day_of_month'] = df.date.dt.day
    df['day_of_year'] = df.date.dt.dayofyear
    df['week_of_year'] = df.date.dt.weekofyear
    df['day_of_week'] = df.date.dt.dayofweek
    df['year'] = df.date.dt.year
    df["is_wknd"] = df.date.dt.weekday // 4
    df['is_month_start'] = df.date.dt.is_month_start.astype(int)
    df['is_month_end'] = df.date.dt.is_month_end.astype(int)
    return df

df = create_date_features(df)
train = create_date_features(train)

train.columns

train.groupby(["store", "item", "month"]).agg({"sales": ["sum", "mean", "median", "std"]})


def random_noise(dataframe):
    return np.random.normal(scale=1.6, size=(len(dataframe),))

df.sort_values(by=['store', 'item', 'date'], axis=0, inplace=True)
train.sort_values(by=["store", "item", "date"], axis=0, inplace=True)


def lag_features(dataframe, lags):
    for lag in lags:
        dataframe['sales_lag_' + str(lag)] = dataframe.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag)) + random_noise(dataframe)
    return dataframe

df = lag_features(df, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])
train = lag_features(train, [91, 98, 105, 112, 119, 126, 182, 364, 546, 728])


def roll_mean_features(dataframe, windows):
    for window in windows:
        dataframe['sales_roll_mean_' + str(window)] = dataframe.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean()) + random_noise(
            dataframe)
    return dataframe

df = roll_mean_features(df, [365, 546])
train = roll_mean_features(train, [365, 546])

def ewm_features(dataframe, alphas, lags):
    for alpha in alphas:
        for lag in lags:
            dataframe['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                dataframe.groupby(["store", "item"])['sales'].transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return dataframe

alphas = [0.95, 0.9, 0.8, 0.7, 0.5]
lags = [91, 98, 105, 112, 180, 270, 365, 546, 728]

df = ewm_features(df, alphas, lags)
train = ewm_features(train, alphas, lags)

check_df(train)

df = pd.get_dummies(df, columns=['store', 'item', 'day_of_week', 'month'])
train = pd.get_dummies(train, columns=['store', 'item', 'day_of_week', 'month'])

check_df(train)

plt.figure(figsize=(10,8))
ax = sns.distplot(x=train["sales"], kde=False, color="orange", hist_kws=dict(edgecolor="black", linewidth=2))
ax.set_title("Distribution of sales")
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(which="both", width=2)
ax.tick_params(which="major", length=7)
ax.tick_params(which="minor", length=4)
plt.show(block=True)

train['sales'] = np.log1p(train["sales"].values)

plt.figure(figsize=(10,8))
ax = sns.distplot(x=train["sales"], kde=False, color="orange", hist_kws=dict(edgecolor="black", linewidth=2))
ax.set_title("Distribution of sales (log)")
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(which="both", width=2)
ax.tick_params(which="major", length=7)
ax.tick_params(which="minor", length=4)
plt.show(block=True)

len(train.loc[train["sales"] <= 2, "sales"]) / len(train)

train.loc[train["sales"] <= 2, "sales"].index

train.drop(train.loc[train["sales"] <= 2, "sales"].index, inplace=True)

plt.figure(figsize=(10,8))
ax = sns.distplot(x=train["sales"], kde=False, color="orange", hist_kws=dict(edgecolor="black", linewidth=2))
ax.set_title("Distribution of sales (log)")
ax.xaxis.set_minor_locator(AutoMinorLocator(2))
ax.yaxis.set_minor_locator(AutoMinorLocator(2))
ax.tick_params(which="both", width=2)
ax.tick_params(which="major", length=7)
ax.tick_params(which="minor", length=4)
plt.show(block=True)



def smape(preds, target):
    n = len(preds)
    masked_arr = ~((preds == 0) & (target == 0))
    preds, target = preds[masked_arr], target[masked_arr]
    num = np.abs(preds - target)
    denom = np.abs(preds) + np.abs(target)
    smape_val = (200 * np.sum(num / denom)) / n
    return smape_val

def lgbm_smape(preds, train_data):
    labels = train_data.get_label()
    smape_val = smape(np.expm1(preds), np.expm1(labels))
    return 'SMAPE', smape_val, False

test

train_ = train.loc[(train["date"] < "2017-01-01"), :]

val = train.loc[(train["date"] >= "2017-01-01") & (train["date"] < "2017-04-01"), :]

cols = [col for col in train.columns if col not in ["date", "id", "sales", "year"]]

y_train = train_["sales"]
X_train = train_[cols]

y_val = val["sales"]
X_val = val[cols]

y_train.shape, X_train.shape, y_val.shape, X_val.shape

import lightgbm as lgb


lgb_full_params = {
    "learning_rate": [0.01, 0.1, 0.2],
    "max_depth": [5, 7, 10, 15],
    "num_leaves": [5, 10, 15],
    "feature_fraction": [0.3, 0.5, 0.7],
    "num_boost_round": 19000,
    "early_stopping_rounds": 200,
    "verbose": 0
}

lgbtrain = lgb.Dataset(data=X_train, label=y_train, feature_name=cols)
lgbval = lgb.Dataset(data=X_val, label=y_val, reference=lgbtrain, feature_name=cols)

def lgb_param_search(lgb_param_dict):
    min_error = float("inf")
    best_params = dict()
    best_iter = float("inf")
    for i in range(len(lgb_param_dict["learning_rate"])):
        lgb_params = dict()
        lgb_params["learning_rate"] = lgb_param_dict["learning_rate"][i]
        for j in range(len(lgb_param_dict["max_depth"])):
            lgb_params["max_depth"] = lgb_param_dict["max_depth"][j]
            for k in range(len(lgb_param_dict["num_leaves"])):
                lgb_params["num_leaves"] = lgb_param_dict["num_leaves"][k]
                for s in range(len(lgb_param_dict["feature_fraction"])):
                    lgb_params["feature_fraction"] = lgb_param_dict["feature_fraction"][s]
                    print(" ")
                    print("##########")
                    print("Learning_rate = " + str(lgb_params["learning_rate"]))
                    print("max_depth = " + str(lgb_params["max_depth"]))
                    print("num_leaves = " + str(lgb_params["num_leaves"]))
                    print("feature_fraction = " + str(lgb_params["feature_fraction"]))
                    model = lgb.train(lgb_params, lgbtrain,
                                      valid_sets=[lgbtrain, lgbval],
                                      num_boost_round=lgb_full_params["num_boost_round"],
                                      early_stopping_rounds=lgb_full_params["early_stopping_rounds"],
                                      feval=lgbm_smape,
                                      verbose_eval=500)
                    print("Learning_rate = " + str(lgb_params["learning_rate"]))
                    print("max_depth = " + str(lgb_params["max_depth"]))
                    print("num_leaves = " + str(lgb_params["num_leaves"]))
                    print("feature_fraction = " + str(lgb_params["feature_fraction"]))
                    if min_error > dict(model.best_score["valid_1"])["SMAPE"]:
                        min_error = dict(model.best_score["valid_1"])["SMAPE"]
                        best_params = model.params
                        best_iter = model.best_iteration
                    else:
                        continue
    return min_error, best_params, best_iter

min_error, best_params, best_iter = lgb_param_search(lgb_full_params)


model = lgb.train(best_params, lgbtrain,
                  valid_sets=[lgbtrain, lgbval],
                  num_boost_round=best_params["num_iterations"],
                  early_stopping_rounds=best_params["early_stopping_round"],
                  feval=lgbm_smape,
                  verbose_eval=1000)

model.params
model.best_iteration
dict(model.best_score["valid_1"])["SMAPE"]
y_pred_val = model.predict(X_val, num_iteration=model.best_iteration)

smape(np.expm1(y_pred_val), np.expm1(y_val))

def plot_lgb_importances(model, plot=False, num=10):
    gain = model.feature_importance('gain')
    feat_imp = pd.DataFrame({'feature': model.feature_name(),
                             'split': model.feature_importance('split'),
                             'gain': 100 * gain / gain.sum()}).sort_values('gain', ascending=False)
    if plot:
        plt.figure(figsize=(10, 10))
        sns.set(font_scale=1)
        sns.barplot(x="gain", y="feature", data=feat_imp[0:25])
        plt.title('feature')
        plt.tight_layout()
        plt.show(block=True)
    else:
        print(feat_imp.head(num))
    return feat_imp

plot_lgb_importances(model, num=30, plot=True)


df['sales'] = np.log1p(df["sales"].values)
test_final = df.loc[df.sales.isna()]
X_test = test_final[cols]

X_train_final = train[cols]
y_train_final = train["sales"]

lgbtrain_all = lgb.Dataset(data=X_train_final, label=y_train_final, feature_name=cols)

final_params = {"learning_rate": 0.01,
              "max_depth": 7,
              "num_leaves": 10,
              "feature_fraction": 0.7,
              'verbose': 0,
              "num_boost_round": model.best_iteration}

final_model = lgb.train(final_params, lgbtrain_all, num_boost_round=model.best_iteration)



test_preds = final_model.predict(X_test, num_iteration=model.best_iteration)

submission_df = test_final.loc[:, ["id", "sales"]]
submission_df["sales"] = np.expm1(test_preds)

submission_df["id"] = submission_df.id.astype(int)

submission_df.to_csv("submission_forecast_2.csv", index=False)