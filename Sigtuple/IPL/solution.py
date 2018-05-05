import pandas as pd
import numpy as np
import joblib


def rename(data):
    """
    renames columns in data
    """
    df = data.copy()
    cols = list(df.columns)
    cols1 = [x for x in cols if (x.lower().find('city') != -1) |
                                (x.lower().find('day') != -1)]

    col_dict = {}
    for col in cols1:
        if col.lower().find('city') != -1:
            col_dict[col] = 'city'
        else:
            col_dict[col] = 'DayOfWeek'

    df.rename(columns=col_dict, inplace=True)
    return df


def combine_datasets(inp_list):
    """
    inp_list: list of dataframes that should be appended
    """
    df = inp_list[0].copy()
    df = rename(df)
    # add sample column
    df['sample'] = 'dev'
    cols = list(df.columns)
    for item in inp_list[1:]:
        data = item.copy()
        data = rename(data)
        data['sample'] = 'val'
        df = pd.concat([df[cols], data[cols]], axis=0)

    print 'shape of combined data:', df.shape
    print df['sample'].value_counts()
    return df


def clean_data(data, out=None):
    """
    data: pandas dataframe (nsamples, nfeatures)
    """
    # remove Game ID, Team 1 and Team 2
    # One Hot Encoding for city
    df = data.copy()
    df1 = pd.get_dummies(df, columns=['city', 'TimeOfGame'],
                         prefix=['city', 'time'], dummy_na=False)
    # date of game
    df1['DateOfGame'] = df1['DateOfGame'].apply(
     lambda x: pd.to_datetime(x, format='%m-%d-%Y'))
    # year, day of month and month
    df1['year'] = df1['DateOfGame'].apply(lambda x: x.year)
    df1['DayOfMonth'] = df1['DateOfGame'].apply(lambda x: x.day)
    df1['month'] = df1['DateOfGame'].apply(lambda x: x.month)

    # dummies for year, month and day
    df2 = pd.get_dummies(df1, columns=['year', 'DayOfMonth', 'month', 'DayOfWeek'],
                         prefix=['year', 'DayOfMonth', 'month', 'DayOfWeek'], dummy_na=False)

    # create prefix for all features
    cols = list(df2.columns)
    for col in ['Game ID', 'Team 1', 'Team 2', 'DateOfGame', 'sample', 'Winner (team 1=1, team 2=0)']:
        cols.remove(col)

    cols1 = ['ST_'+x.lower() for x in cols]
    col_dict = dict(zip(cols, cols1))
    df2.rename(columns=col_dict, inplace=True)
    df2.rename(columns={'Winner (team 1=1, team 2=0)': 'dv', 'Game ID': 'index'}, inplace=True)
    df2.reset_index(drop=True, inplace=True)

    # convert dv to categorical dtype
    df2['dv'] = df2['dv'].astype(float)

    if out:
        df2.to_csv(out, index=False)
    else:
        return df2


def duplicate_sample(data):
    df = data.copy()
    dev = df[df['sample'] == 'dev']
    dev.reset_index(drop=True, inplace=True)
    oot = dev.copy()
    oot['sample'] = 'oot'
    cols = list(df.columns)
    final = pd.concat([df[cols], oot[cols]], axis=0)
    final.reset_index(drop=True, inplace=True)
    return final


def preprocessing(data):
    """
    missing value imputation
    outlier treatment
    scaling
    """
    # imputation
    from kbglearn.preprocess.preprocessing import MaxAUCImputerCV
    dev = data.copy()
    dev.reset_index(drop=True, inplace=True)
    cols = [x for x in list(dev.columns) if x.startswith('ST')]
    x_dev = dev[cols].values
    y_dev = dev['dv'].values
    imputer = MaxAUCImputerCV()
    imputer.fit(x_dev, y_dev)
    data_imputed = imputer.transform(data[cols].values)
    data_imputed = pd.DataFrame(data_imputed, columns=cols)
    # add back index, sample and target
    data1 = pd.concat([data_imputed, data[['index', 'dv', 'sample']]], axis=1)

    # outlier treatment
    from kbglearn.preprocess.preprocessing import LegacyOutlierScaler
    dev = data1.copy()
    dev.reset_index(drop=True, inplace=True)
    x_dev = dev[cols].values
    outlier = LegacyOutlierScaler()
    outlier.fit(x_dev)
    data_outlier = outlier.transform(data[cols].values)
    data_outlier = pd.DataFrame(data_outlier, columns=cols)
    # add back index, sample and target
    data2 = pd.concat([data_outlier, data[['index', 'dv', 'sample']]], axis=1)

    # scaling
    from sklearn.preprocessing import StandardScaler
    dev = data2.copy()
    dev.reset_index(drop=True, inplace=True)
    x_dev = dev[cols].values
    scaler = StandardScaler()
    scaler.fit(x_dev)
    data_scaled = scaler.transform(data[cols].values)
    data_scaled = pd.DataFrame(data_scaled, columns=cols)
    # add back index, sample and target
    data2 = pd.concat([data_scaled, data[['index', 'dv', 'sample']]], axis=1)

    data2.reset_index(drop=True, inplace=True)

    return imputer, outlier, scaler, data2


if __name__ == '__main__':
    print 'model is running ...'
    # read the datasets
    path = '/Users/vnathan/Documents/ExternalTests/Sigtuple/IPL/'
    train = pd.read_csv(path + 'train.csv')
    test = pd.read_csv(path + 'test.csv')
    inp = [train, test]
    # append train and test data
    df = combine_datasets(inp)
    df1 = clean_data(df)
    df2 = duplicate_sample(df1)
    df2.to_csv(path+'data_prepared.csv', index=False)

    # preprocessing
    imputer, outlier, scaler, df3 = preprocessing(df2)
    imputer.dump(open(''), 'w')
