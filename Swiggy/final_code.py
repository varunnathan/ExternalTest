import pandas as pd
import numpy as np
import json
import pickle
from dateutil.relativedelta import relativedelta
from sklearn.cross_validation import train_test_split
from sklearn import cross_validation, metrics
from sklearn.linear_model import Lasso, Ridge
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import r2_score


# globals
path = '/Users/vnathan/Documents/ExternalTests/SW/'
inp_file = 'data.csv'
cust_id = 'customer_id'
dv1 = 'Amount in last 7 days'
dv2 = 'Avg. Amount per order in last 7 days'
date_cols = ['First Time', 'Recent Time']


def impute_missing(data, feat_cols):
    """
    imputes missing values in feat_cols with zero
    data: dataframe with shape (nsamples, nfeatures)
    feat_cols: list of feature columns
    """
    df = data.copy()
    for col in feat_cols:
        print col
        df[col] = df[col].fillna(0)
    return df


def calc_ratios(data, num, denom, impute_val, ratio_col):
    df = data.copy()
    mask = (df[num].notnull()) & (df[denom].notnull()) & (df[denom] != 0)
    df.loc[mask, ratio_col] = map(
     lambda x, y: 1.*x/y, df.loc[mask, num], df.loc[mask, denom])
    df.loc[~mask, ratio_col] = impute_val
    return df[ratio_col]


def data_prep(data):
    df = data.copy()
    df['Recent Time'] = df['Recent Time'].apply(
     lambda x: x - relativedelta(days=7))
    df['# of Orders'] = df['# of Orders'] - df['# of Orders in last 7 days']
    df['# of Orders in last 3 weeks'] = (df['# of Orders in last 4 weeks'] -
                                         df['# of Orders in last 7 days'])
    df['Amount'] = df['Amount'] - df['Amount in last 7 days']
    df['Amount in last 3 weeks'] = (df['Amount in last 4 weeks'] -
                                    df['Amount in last 7 days'])
    return df


def feat_engg(data, prefix='SW_'):
    df = data.copy()
    df[prefix+'MOB'] = map(
     lambda x, y: relativedelta(y, x).months, df['First Time'],
     df['Recent Time'])
    df[prefix+'DOB'] = map(
     lambda x, y: relativedelta(y, x).days, df['First Time'],
     df['Recent Time'])
    df[prefix+'num_Orders_per_month'] = calc_ratios(
     df, 'SW_# of Orders', 'SW_MOB', impute_val=np.NaN,
     ratio_col=prefix+'num_Orders_per_month')
    df[prefix+'num_Orders_per_week'] = df[
     'SW_# of Orders in last 3 weeks'].apply(lambda x: 1.*x/3)
    df[prefix+'num_Orders_prior_last_3_weeks'] = (
     df['SW_# of Orders'] - df['SW_# of Orders in last 3 weeks'])
    df[prefix+'accel_num_orders1'] = calc_ratios(
     df, 'SW_# of Orders in last 3 weeks', 'SW_# of Orders', impute_val=0,
     ratio_col=prefix+'accel_num_orders1')
    df[prefix+'accel_num_orders2'] = calc_ratios(
     df, 'SW_# of Orders in last 3 weeks', 'SW_num_Orders_prior_last_3_weeks',
     impute_val=np.NaN, ratio_col=prefix+'accel_num_orders2')
    df[prefix+'avg_amt_per_month'] = calc_ratios(
     df, 'SW_Amount', 'SW_MOB', impute_val=np.NaN,
     ratio_col=prefix+'avg_amt_per_month')
    df[prefix+'avg_amt_per_week'] = df[
     'SW_Amount in last 3 weeks'].apply(lambda x: 1.*x/3)
    df[prefix+'avg_amt_per_order'] = calc_ratios(
     df, 'SW_Amount', 'SW_# of Orders', impute_val=0,
     ratio_col=prefix+'avg_amt_per_order')
    df[prefix+'avg_amt_per_order_last_3_weeks'] = calc_ratios(
     df, 'SW_Amount in last 3 weeks', 'SW_# of Orders in last 3 weeks',
     impute_val=0, ratio_col=prefix+'avg_amt_per_order_last_3_weeks')
    df[prefix+'amt_prior_last_3_weeks'] = (
     df['SW_Amount'] - df['SW_Amount in last 3 weeks'])
    df[prefix+'accel_amt1'] = calc_ratios(
     df, 'SW_Amount in last 3 weeks', 'SW_Amount', impute_val=0,
     ratio_col=prefix+'accel_amt1')
    df[prefix+'accel_amt2'] = calc_ratios(
     df, 'SW_Amount in last 3 weeks', 'SW_amt_prior_last_3_weeks',
     impute_val=np.NaN, ratio_col=prefix+'accel_amt2')
    df[prefix+'total_dist_from_rest'] = map(
     lambda x, y: 1.*x*y, df['SW_Avg_DistanceFromResturant'],
     df['SW_# of Orders'])
    df[prefix+'total_dist_from_rest_last_3_weeks'] = map(
     lambda x, y: 1.*x*y, df['SW_Avg_DistanceFromResturant'],
     df['SW_# of Orders in last 3 weeks'])
    df[prefix+'total_dist_from_rest_prior_last_3_weeks'] = (
     df[prefix+'total_dist_from_rest'] -
     df[prefix+'total_dist_from_rest_last_3_weeks'])
    df[prefix+'accel_total_dist_from_rest1'] = calc_ratios(
     df, 'SW_total_dist_from_rest_last_3_weeks', 'SW_total_dist_from_rest',
     impute_val=0, ratio_col=prefix+'accel_total_dist_from_rest1')
    df[prefix+'accel_total_dist_from_rest2'] = calc_ratios(
     df, 'SW_total_dist_from_rest_last_3_weeks',
     'SW_total_dist_from_rest_prior_last_3_weeks', impute_val=np.NaN,
     ratio_col=prefix+'accel_total_dist_from_rest2')
    df[prefix+'total_deliv_time'] = map(
     lambda x, y: 1.*x*y, df['SW_Avg_DeliveryTime'], df['SW_# of Orders'])
    df[prefix+'total_deliv_time_last_3_weeks'] = map(
     lambda x, y: 1.*x*y, df['SW_Avg_DeliveryTime'],
     df['SW_# of Orders in last 3 weeks'])
    df[prefix+'total_deliv_time_prior_last_3_weeks'] = (
     df[prefix+'total_deliv_time'] -
     df[prefix+'total_deliv_time_last_3_weeks'])
    df[prefix+'accel_total_deliv_time1'] = calc_ratios(
     df, 'SW_total_deliv_time_last_3_weeks', 'SW_total_deliv_time',
     impute_val=0, ratio_col=prefix+'accel_total_deliv_time1')
    df[prefix+'accel_total_deliv_time2'] = calc_ratios(
     df, 'SW_total_deliv_time_last_3_weeks',
     'SW_total_deliv_time_prior_last_3_weeks', impute_val=np.NaN,
     ratio_col=prefix+'accel_total_deliv_time2')
    df[prefix+'avg_dist_from_rest_ratio_avg_deliv_time'] = calc_ratios(
     df, 'SW_Avg_DistanceFromResturant', 'SW_Avg_DeliveryTime',
     impute_val=np.NaN,
     ratio_col=prefix+'avg_dist_from_rest_ratio_avg_deliv_time')
    mask = df['SW_Avg_DistanceFromResturant'] < 0
    df.loc[mask, prefix+'avg_dist_from_rest_less_than_zero_flag'] = 1
    df.loc[~mask, prefix+'avg_dist_from_rest_less_than_zero_flag'] = 0
    # date variables
    df['year'] = df['Recent Time'].apply(lambda x: x.year)
    df['month'] = df['Recent Time'].apply(lambda x: x.month)
    df['day'] = df['Recent Time'].apply(lambda x: x.day)
    df['dayofweek'] = df['Recent Time'].apply(lambda x: x.dayofweek)
    df['dayofyear'] = df['Recent Time'].apply(lambda x: x.dayofyear)
    return df


def sample_split(data, split=0.3, random_state=18112017):
    df = data.copy()
    features = [x for x in list(df.columns) if x != dv1]
    x_train, x_test, y_train, y_test = train_test_split(
     df[features], df[dv1], test_size=split,
     random_state=random_state)
    # dev and val samples
    dev = pd.concat([x_train, y_train], axis=1)
    dev['sample'] = 'dev'
    val = pd.concat([x_test, y_test], axis=1)
    val['sample'] = 'val'
    df1 = pd.concat(
     [dev[features+['sample', dv1]], val[features+['sample', dv1]]], axis=0)
    df1.reset_index(drop=True, inplace=True)
    return df1


def count_vars(data, cat_col):
    df = data.copy()
    mask = df['sample'] == 'dev'
    dev = df.loc[mask, :]
    dev.reset_index(drop=True, inplace=True)
    agg_fn = {dv1: np.mean, 'SW_Amount in last 3 weeks': np.mean}
    table = pd.DataFrame(dev.groupby(cat_col).agg(agg_fn))
    table.reset_index(inplace=True)
    table['ratio'] = map(
     lambda x, y: 1.*x/y, table[dv1], table['SW_Amount in last 3 weeks'])
    mask = table['ratio'].isnull()
    table.loc[mask, 'ratio'] = 0
    print table
    dv_dict = dict(zip(table[cat_col], table[dv1]))
    ratio_dict = dict(zip(table[cat_col], table['ratio']))
    tmp1 = df[cat_col].map(dv_dict)
    tmp2 = map(
     lambda x, y: ratio_dict[x]*y, df[cat_col],
     df['SW_Amount in last 3 weeks'])
    return tmp1, tmp2


def preprocess(data):
    """
    imputation of missing values
    """
    df = data.copy()
    dev = df[df['sample'] == 'dev']
    dev.reset_index(drop=True, inplace=True)
    features = [x for x in list(df.columns) if x.startswith('SW_')]
    miss_val = []
    for col in features:
        value = dev[col].median()
        miss_val.append(value)

    miss_dict = dict(zip(features, miss_val))

    df.fillna(miss_dict, inplace=True)

    return miss_dict, df


def rmse(target, predict):
    return (np.mean((target - predict) ** 2)) ** 0.5


def perform_gridsearch(data, alg, params, scoring, cv_fold):
    df = data.copy()
    dev = df[df['sample'] == 'dev']
    dev.reset_index(drop=True, inplace=True)
    val = df[df['sample'] == 'val']
    val.reset_index(drop=True, inplace=True)
    features = [x for x in list(df.columns) if x.startswith('SW_')]
    print len(features)
    x_dev = dev[features].values
    y_dev = dev[dv1].values
    y_dev = y_dev.reshape((y_dev.shape[0], 1))
    x_val = val[features].values
    y_val = val[dv1].values
    y_val = y_val.reshape((y_val.shape[0], 1))
    grid_model = GridSearchCV(alg, params, scoring=scoring, cv=cv_fold)
    grid_model.fit(x_dev, y_dev)
    val_pred = grid_model.predict(x_val)
    print 'best params: ', grid_model.best_params_
    print 'RMSE: ', rmse(y_val, val_pred)
    print 'R2 Score: ', r2_score(y_val, val_pred)


def modelfit(alg, dtrain, dtest, predictors, target, performCV=True,
             cv_folds=4):
    # Fit the algorithm on the data
    print 'fit'
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_pred = alg.predict(dtrain[predictors])

    # Predict on testing set:
    dtest_pred = alg.predict(dtest[predictors])

    if performCV:
        print 'Perform cross-validation:'
        cv_score = cross_validation.cross_val_score(
         alg, dtrain[predictors], dtrain[target], cv=cv_folds,
         scoring='r2')

    # Print model report:
    print "\nModel Report"
    print "Dev AUC Score : %f" % metrics.r2_score(
     dtrain[target], dtrain_pred)
    print "Val AUC Score : %f" % metrics.r2_score(
     dtest[target], dtest_pred)

    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))
    return alg, dtrain_pred, dtest_pred


def get_cuts(data, bins, target):
    df = data.copy()
    table = df.groupby(bins)[target].min()
    cuts = table.unique().tolist()
    cuts.sort()
    return cuts


def apply_cuts(data, cuts, target):
    df = data.copy()
    temp = df[target]
    binned = pd.Series([-2] * len(temp), index=temp.index)
    binned[temp.isnull()] = -1
    binned[temp < np.min(cuts)] = 0

    for ibin, (low, high) in enumerate(zip(cuts[:-1], cuts[1:])):
        mask = (temp >= low) & (temp < high)
        binned[mask] = ibin + 1
    binned[temp >= np.max(cuts)] = len(cuts)
    return binned


if __name__ == '__main__':
    print 'prediction of customer spend in the next 7 days...'
    # read dataset
    df = pd.read_csv(path+inp_file)
    df1 = impute_missing(df, feat_cols=[
     '# of Orders in last 7 days', '# of Orders in last 4 weeks'])
    df1[dv2] = calc_ratios(df1, num=dv1, denom='# of Orders in last 7 days',
                           impute_val=0, ratio_col=dv2)
    for col in date_cols:
        print col
        df1[col] = df1[col].apply(pd.to_datetime)

    df2 = data_prep(df1)
    # add prefix to feat_cols
    feat_cols = ['# of Orders', 'Amount', 'Avg_DistanceFromResturant',
                 'Avg_DeliveryTime', '# of Orders in last 3 weeks',
                 'Amount in last 3 weeks']
    feat_cols_new = ['SW_'+x for x in feat_cols]
    df2.rename(columns=dict(zip(feat_cols, feat_cols_new)), inplace=True)
    # feature engineering
    df3 = feat_engg(df2)
    # sample split
    df4 = sample_split(df3)
    # feature engineering with month, day and dayofweek
    df4['SW_mean_amt_month'], df4['SW_ratio_mean_amt_month'] = count_vars(
     df4, cat_col='month')
    df4['SW_mean_amt_day'], df4['SW_ratio_mean_amt_day'] = count_vars(
     df4, cat_col='day')
    df4['SW_mean_amt_dayofweek'], df4[
     'SW_ratio_mean_amt_dayofweek'] = count_vars(df4, cat_col='dayofweek')

    # impute missing values
    miss_dict, df5 = preprocess(df4)
    # save
    json.dump(miss_dict, open(path+'imputation_values.json', 'w'))
    df5.to_csv(path+'final_modelling_data.csv', index=False)

    # gridsearch and modelling
    # lasso
    lasso_model = Lasso()
    alpha = [0.001, 0.005, 0.01, 0.03, 0.1, 0.3, 0.5, 0.6, 0.7, 1]
    params = {'alpha': alpha}
    scoring = 'r2'
    perform_gridsearch(df5, lasso_model, params, scoring, cv_fold=4)
    '''
    best params:  {'alpha': 0.7}
    RMSE:  481.61811491
    R2 Score:  0.376216707187
    '''
    # ridge
    ridge_model = Ridge()
    alpha = [400, 300, 200, 100, 30, 10, 5, 3]
    params = {'alpha': alpha}
    scoring = 'r2'
    perform_gridsearch(df5, ridge_model, params, scoring, cv_fold=4)
    '''
    best params:  {'alpha': 300}
    RMSE:  330.634275645
    R2 Score:  0.376043668689
    '''
    # GBR
    gbr = GradientBoostingRegressor()
    SEED = 54545454
    params = {
        "max_depth": [2, 3, 4],
        "n_estimators": [250, 500, 1000, 2000],
        "min_samples_split": [10, 30],
        "max_features": ['sqrt'],
        "learning_rate": [0.01, 0.03, 0.1],
        "subsample": [0.7, 0.9, 1.0],
        "min_samples_leaf": [10, 30],
        'random_state': [SEED],
    }
    scoring = 'r2'
    perform_gridsearch(df5, gbr, params, scoring, cv_fold=4)
    '''
    best params:  {'subsample': 0.7, 'learning_rate': 0.01,
    'min_samples_leaf': 10, 'n_estimators': 500, 'min_samples_split': 10,
    'random_state': 54545454, 'max_features': 'sqrt', 'max_depth': 4}
    RMSE:  479.658316447
    R2 Score:  0.44472400572
    '''
    # build final model - train on overall dev
    mask = df5['sample'] == 'dev'
    dev = df5.loc[mask, :]
    val = df5.loc[~mask, :]
    dev.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    predictors = [x for x in list(df5.columns) if x.startswith('SW_')]
    gbm_tuned = GradientBoostingRegressor(
     learning_rate=0.01, n_estimators=500,
     max_depth=4, min_samples_split=10, min_samples_leaf=10, subsample=0.7,
     random_state=54545454, max_features='sqrt')
    model, dev_pred, val_pred = modelfit(
     gbm_tuned, dev, val, predictors, target=dv1, performCV=True)
    '''
    Model Report
    Dev R2 Score : 0.515522
    Val R2 Score : 0.444724
    CV Score : Mean - 0.42 | Std - 0.02704939 | Min - 0.35 | Max - 0.45
    '''
    df5['pred_{}'.format(dv1)] = model.predict(df5[predictors])
    # save model
    pickle.dump(model, open(path+'final_regression_model.pkl', 'w'))
    # segment customers based on predicted scores
    mask = df5['sample'] == 'dev'
    dev = df5.loc[mask, :]
    dev.reset_index(drop=True, inplace=True)
    dev['pred_bins'] = pd.qcut(dev['pred_{}'.format(dv1)], 10, labels=False)
    cuts = get_cuts(dev, 'pred_bins', 'pred_{}'.format(dv1))
    df5['pred_bins'] = apply_cuts(df5, cuts, 'pred_{}'.format(dv1))
    mask = df5['pred_bins'] == 0
    df5.loc[mask, 'pred_bins'] = 1
    df5.to_csv(path+'prediction_from_final_model.csv', index=False)
    # feature importance
    feat_imp = pd.DataFrame(
     {'Feature': predictors, 'Importance': model.feature_importances_})
    feat_imp.sort('Importance', ascending=False, inplace=True)
    feat_imp.to_csv(path+'feat_imp.csv', index=False)
