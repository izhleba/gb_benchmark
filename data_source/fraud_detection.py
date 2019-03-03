import pandas as pd
import gc

def timeFeatures(df):
    # Make some new features with click_time column
    df['datetime'] = pd.to_datetime(df['click_time'])
    df['dow']      = df['datetime'].dt.dayofweek
    df["doy"]      = df["datetime"].dt.dayofyear
    #df["dteom"]    = df["datetime"].dt.daysinmonth - df["datetime"].dt.day
    df.drop(['click_time', 'datetime'], axis=1, inplace=True)
    return df

def data_fraud_detection(num_rows):
    train_columns = ['ip', 'app', 'device', 'os', 'channel', 'click_time', 'is_attributed']
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'uint32'
    }

    train = pd.read_csv("data/talkingdata-adtracking-fraud-detection/train.csv", nrows=num_rows, usecols=train_columns, dtype=dtypes)
    y = train['is_attributed']
    train.drop(['is_attributed'], axis=1, inplace=True)

    # Count the number of clicks by ip
    ip_count = train.groupby(['ip'])['channel'].count().reset_index()
    ip_count.columns = ['ip', 'clicks_by_ip']
    train = pd.merge(train, ip_count, on='ip', how='left', sort=False)
    train['clicks_by_ip'] = train['clicks_by_ip'].astype('uint16')
    train.drop('ip', axis=1, inplace=True)
    train = timeFeatures(train)
    gc.collect()
    return train,y
