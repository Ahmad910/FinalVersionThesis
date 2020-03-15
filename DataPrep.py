import pyodbc
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timedelta

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Alter the metaLibraryKey [39, 40, 454, 743] to generate csv file
metaLibraryKey = '743'

if metaLibraryKey == '743' or metaLibraryKey == '454':
    database = 'METADATA'
else:
    database = 'METADATA_FD'

# change the Server name as the server name you have in SSMS
conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=DESKTOP-0OC51CA\SQLEXPRESS;"
                      "Database=" + database + ";"
                      "Trusted_Connection=yes;")

SQL_Query = pd.read_sql_query(
    "SELECT * FROM MyView WHERE MetaLibraryKey = " + metaLibraryKey + " AND StatusMainProcess = 'SUCCESS' ", conn)

df_raw = pd.DataFrame(SQL_Query)
df_raw['MetaLibraryKey'] = pd.to_numeric(df_raw['MetaLibraryKey'], errors='coerce')
df_raw['DurationSecondsMainProcess'] = pd.to_numeric(df_raw['DurationSecondsMainProcess'], errors='coerce')
df_raw['DurationSecondsSubProcess'] = pd.to_numeric(df_raw['DurationSecondsSubProcess'], errors='coerce')
df_raw['InsertSum'] = pd.to_numeric(df_raw['InsertSum'], errors='raise')
df_raw['SourceSum'] = pd.to_numeric(df_raw['SourceSum'], errors='raise')
df_raw['DeleteSum'] = pd.to_numeric(df_raw['DeleteSum'], errors='raise')
df_raw['UpdateSum'] = pd.to_numeric(df_raw['UpdateSum'], errors='raise')
df_raw['ExecutionDay'] = pd.to_numeric(df_raw['ExecutionDay'], errors='raise')
# Cast string to datetime
df_raw['StartMainProcess'] = pd.to_datetime(df_raw['StartMainProcess'])
df_raw['EndMainProcess'] = pd.to_datetime(df_raw['EndMainProcess'])
date1 = df_raw['StartMainProcess'].iloc[0]
date2 = df_raw['StartMainProcess'].iloc[-1]
df_raw = df_raw.set_index('StartMainProcess')
df = pd.DataFrame()
df['MetaLibraryKey'] = df_raw.MetaLibraryKey.resample('D').mean()
df['MetaProcMasterKey'] = df_raw.MetaProcMasterKey.resample('D').first()
df['DurationSecondsMainProcess'] = df_raw.DurationSecondsMainProcess.resample('D').mean()
df['InsertSum'] = df_raw.InsertSum.resample('D').mean()
df['ExecutionDay'] = df_raw.ExecutionDay.resample('D').first()
df['ExecutionMonth'] = df_raw.ExecutionMonth.resample('D').first()
my_dates = pd.date_range(date1, date2 + timedelta(days=1)).tolist()
months = list()
weekdays = list()
days = list()

for item in my_dates:
    months.append(item.strftime("%B"))
    weekdays.append(datetime.weekday(item))
    days.append(item.day)
df['ExecutionMonth'] = months
df['ExecutionWeekDay'] = weekdays
df['ExecutionDay'] = days
df = df.reset_index()
df = df.bfill()
df = df.reset_index(drop=True)

if metaLibraryKey == '743':
    start = '2017-05-21'
    end = '2019-06-10'
elif metaLibraryKey == '39':
    start = '2016-11-18'
    end = '2017-11-17'
elif metaLibraryKey == '454':
    start = '2016-06-01'
    end = '2017-05-17'
else:
    start = '2017-11-20'
    end = '2018-10-28'

mask = (df['StartMainProcess'] > start) & (df['StartMainProcess'] <= end)
df = df.loc[mask]
df.reset_index(inplace=True, drop=True)
df_clone = df
df_copy = df_clone

def normalize(feature_to_normalize):
    df[feature_to_normalize] = df[feature_to_normalize].astype(int)
    durationMain = df[feature_to_normalize].values.reshape(-1, 1)
    duration_scaler = MinMaxScaler()
    durationMain_scaled = duration_scaler.fit_transform(durationMain)
    durationMain_scaled_df = pd.DataFrame(durationMain_scaled)
    df[feature_to_normalize] = durationMain_scaled_df[0]
    df_copy[feature_to_normalize] = durationMain_scaled_df[0]

normalize('DurationSecondsMainProcess')
normalize('InsertSum')

df_labels = df['DurationSecondsMainProcess']
columnsToDropOneHot = ['MetaLibraryKey', 'MetaProcMasterKey', 'ExecutionDay',
                       'ExecutionWeekDay', 'ExecutionMonth', 'StartMainProcess']

df.drop(columnsToDropOneHot, inplace=True, axis=1)
df.reset_index(inplace=True, drop=True)
df.to_csv(r'C:\Users\ahmad\Desktop\Thesis\autoencoder_' + metaLibraryKey + 'MS.csv', index=False)
df.drop(['DurationSecondsMainProcess'], inplace=True, axis=1)
df.to_csv(r'C:\Users\ahmad\Desktop\Thesis\Prediction_' + metaLibraryKey + '_MS.csv', index=False)
df_labels.to_csv(r'C:\Users\ahmad\Desktop\Thesis\labels_' + metaLibraryKey + '_MS.csv', index=False)
