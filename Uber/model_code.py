import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from math import log
from utility import MaxAUCImputerCV, LegacyOutlierScaler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.cross_validation import train_test_split, cross_val_score

# GLOBALS
inp_path = '/Users/vnathan/Documents/ExternalTests/Uber/'
inp = 'ub_dataset.csv'
pipe = 'pipeline.pkl'
pre = 'ub_dataset_preprocessed.pkl'
dv = 'DV'
index = 'id'
date_cols = ['first_completed_date', 'signup_date', 'bgc_date',
             'vehicle_added_date']
flag_cols = ['bgc_date', 'vehicle_added_date']
dummy_cols = ['city_name', 'signup_os', 'signup_channel', 'vehicle_make',
              'vehicle_model']
prefix = 'MI'
PREPROCESS = {
    'exoutscaler': LegacyOutlierScaler(),
    'exaucimputer': MaxAUCImputerCV(),
    'stdscaler': StandardScaler()
}
steps = ['exaucimputer', 'exoutscaler', 'stdscaler']


def num_days(d1, d2):
    """
    number of days between d1 and d2
    d1: series in datetime format
    d2: series in datetime format
    """
    mask = (d1.notnull()) & (d2.notnull())
    d = pd.Series(len(d1)*[None])
    d[mask] = map(lambda x, y: (x-y).days, d2[mask], d1[mask])
    d = d.astype(float)
    return d


def date_comp(d):
    """
    returns year, month, day of week, week of year
    """
    mask = d.notnull()
    year = pd.Series(len(d)*[None])
    year[mask] = d[mask].apply(lambda x: x.year)
    year = year.astype(float)
    month = pd.Series(len(d)*[None])
    month[mask] = d[mask].apply(lambda x: x.month)
    month = month.astype(float)
    dow = pd.Series(len(d)*[None])
    dow[mask] = d[mask].apply(lambda x: x.dayofweek)
    dow = dow.astype(float)
    woy = pd.Series(len(d)*[None])
    woy[mask] = d[mask].apply(lambda x: x.weekofyear)
    woy = woy.astype(float)
    return [year, month, dow, woy]


def get_flag(series):
    return series.apply(lambda x: 1 if pd.notnull(x) else 0)


def get_woe(x, y):
    table = pd.crosstab(x, y)
    table.reset_index(inplace=True)
    table.rename(columns={0: '#good', 1: '#bad'}, inplace=True)
    total_good = table['#good'].sum()
    total_bad = table['#bad'].sum()
    table['perc_good'] = table['#good'].apply(lambda x: x/float(total_good))
    table['perc_bad'] = table['#bad'].apply(lambda x: x/float(total_bad))
    mask = (table['perc_good'] != 0) & (table['perc_bad'] != 0)
    table.loc[mask, 'WOE'] = map(
     lambda x, y: log(x / float(y)), table.loc[mask, 'perc_good'],
     table.loc[mask, 'perc_bad'])
    table.loc[~mask, 'WOE'] = np.NaN
    table.reset_index(drop=True, inplace=True)
    return table


def impute_nulls_cat_vars(data, feat_cols):
    df = data.copy()
    for col in feat_cols:
        print col
        mask = df[col].isnull()
        df.loc[mask, col] = 'missing'
        print pd.crosstab(df[col], df[dv])

    return df


def sample_split(data, split=0.3, random_state=27012018):
    df = data.copy()
    features = [x for x in list(df.columns) if x.startswith(prefix)]
    x_train, x_test, y_train, y_test = train_test_split(
     df[features+['id']], df[dv], test_size=split,
     random_state=random_state)
    # dev and val samples
    dev = pd.concat([x_train, y_train], axis=1)
    dev['sample'] = 'dev'
    val = pd.concat([x_test, y_test], axis=1)
    val['sample'] = 'val'
    cols = features + [dv, 'sample', 'id']
    df1 = pd.concat([dev[cols], val[cols]], axis=0)
    df1.reset_index(drop=True, inplace=True)
    return df1


def preprocess(data, steps):
    """
    imputation, outlier treatment and scaling
    """
    df = data.copy()
    other_cols = ['sample', 'id']
    features = [x for x in list(df.columns) if x.startswith(prefix)]
    classic_steps = steps
    steps = list(zip(steps, map(PREPROCESS.get, steps)))
    datapipe = Pipeline(steps=steps)
    # dev data
    dev = df[df['sample'] == 'dev']
    dev.reset_index(drop=True, inplace=True)
    # remove features that are completely missing in dev
    feats_all_missing = [
     x for x in features if dev[x].isnull().sum() == dev.shape[0]]
    print 'removed %d features as they are completely missing in dev:' % (len(
     feats_all_missing))
    features = list(set(features) - set(feats_all_missing))
    x_dev = dev[features].values
    y_dev = dev[dv].values.astype(float)
    print 'fit'
    datapipe.fit(x_dev, y_dev)
    print 'transform dataframe using pipeline'
    print 'train data:'
    train = datapipe.transform(df[features].values)
    train = pd.DataFrame(train, columns=features)
    train = pd.concat([train, df[other_cols+[dv]]], axis=1)
    # Create "classic" datapipe and store list of features
    classic_pipe = Pipeline([(name, datapipe.named_steps[name])
                             for name in classic_steps])
    classic_pipe.feature_names = features

    return train, classic_pipe


def modelfit(alg, dtrain, dtest, predictors, target, performCV=True,
             cv_folds=4):
    # Fit the algorithm on the data
    print 'fit'
    alg.fit(dtrain[predictors], dtrain[target])

    # Predict training set:
    dtrain_predprob = alg.predict_proba(dtrain[predictors])[:, 1]

    # Predict on testing set:
    dtest_predprob = alg.predict_proba(dtest[predictors])[:, 1]

    if performCV:
        print 'Perform cross-validation:'
        cv_score = cross_val_score(
         alg, dtrain[predictors], dtrain[target], cv=cv_folds,
         scoring='roc_auc')

    # Print model report:
    print "\nModel Report"
    print "Dev AUC Score : %f" % roc_auc_score(dtrain[target], dtrain_predprob)
    print "Val AUC Score : %f" % roc_auc_score(dtest[target], dtest_predprob)
    print "Dev Log Loss : %f" % log_loss(dtrain[target], dtrain_predprob)
    print "Val Log Loss : %f" % log_loss(dtest[target], dtest_predprob)

    if performCV:
        print "CV Score : Mean - %.7g | Std - %.7g | Min - %.7g | Max - %.7g" % (np.mean(cv_score), np.std(cv_score), np.min(cv_score), np.max(cv_score))
    return alg, dtrain_predprob, dtest_predprob


if __name__ == '__main__':
    print 'prediction of driving given signups'
    df = pd.read_csv(inp_path+inp)
    # DV
    df[dv] = df['first_completed_date'].apply(
     lambda x: 1 if pd.notnull(x) else 0)
    # convert date_cols to datetime format
    for col in date_cols:
        print col
        df[col] = df[col].apply(lambda x: pd.to_datetime(x, format='%m/%d/%y'))
    d1 = 'signup_date'
    name1 = d1.split('_')[0]
    for date in ['bgc_date', 'vehicle_added_date']:
        name2 = date.split('_')[0]
        name = 'num_days_from_'+name1+'_to_'+name2
        print name
        df[name] = num_days(df[d1], df[date])
    df['year_signup'], df['month_signup'], df['day_of_week_signup'], df[
     'week_of_year_signup'] = date_comp(df['signup_date'])
    df1 = df.copy()
    mask = df1['vehicle_year'] == 0
    df1.loc[mask, 'vehicle_year'] = None
    mask = df1['vehicle_year'].notnull()
    df1.loc[mask, 'num_years_from_vehicle_made'] = df1.loc[
     mask, 'vehicle_year'].apply(lambda x: 2016 - x)
    # flag columns
    for col in flag_cols:
        print col
        df1[col+'flag'] = get_flag(df1[col])
    # impute nulls in categorical vars with 'missing'
    df1 = impute_nulls_cat_vars(df1, feat_cols=dummy_cols+['vehicle_year'])
    for col in dummy_cols+['vehicle_year']:
        print col
        table = get_woe(df1[col], df1[dv])
        map_dict = dict(zip(table[col], table['WOE']))
        print map_dict
        df1[col] = df1[col].map(map_dict)

    # remove date columns
    df2 = df1.copy()
    df2.drop(date_cols, axis=1, inplace=True)

    # add prefix
    cols = list(df2.columns)
    feat_cols = [x for x in cols if x not in ['id', dv]]
    new_feat_cols = [prefix+'_'+x for x in feat_cols]
    df2.rename(columns=dict(zip(feat_cols, new_feat_cols)), inplace=True)

    # sampling
    df3 = sample_split(df2)

    # preprocessing
    df_pre, pipeline = preprocess(df3, steps=steps)

    df_pre['uid'] = range(len(df_pre))

    # save
    df_pre.to_pickle(inp_path+pre)
    pickle.dump(pipeline, open(inp_path+pipe, 'w'))

    # gridsearch
    mask = df_pre['sample'] == 'dev'
    dev = df_pre.loc[mask, :]
    val = df_pre.loc[~mask, :]
    dev.reset_index(drop=True, inplace=True)
    val.reset_index(drop=True, inplace=True)
    predictors = [x for x in list(df_pre.columns) if x.startswith(prefix)]
    gbm_tuned = GradientBoostingClassifier(
     learning_rate=0.01, n_estimators=2000, loss='deviance',
     max_depth=4, min_samples_split=30, min_samples_leaf=30, subsample=1.0,
     random_state=54545454, max_features='sqrt')
    model, dev_prob, val_prob = modelfit(
     gbm_tuned, dev, val, predictors, target=dv, performCV=True)

    # save model object
    pickle.dump(model, open(inp_path+'gbc_best.pkl', 'w'))

    # score samples
    cols_needed = ['id', dv]
    dev['pred_prob'] = dev_prob
    val['pred_prob'] = val_prob
    df_pre['pred_prob'] = model.predict_proba(df_pre[predictors])[:, 1]

    # get feature importance
    feat_imp = pd.DataFrame(
     {'feature': predictors, 'importance': model.feature_importances_})
    feat_imp.sort('importance', ascending=False, inplace=True)
    # save
    feat_imp.to_csv(inp_path+'feature_importance.csv', index=False)
