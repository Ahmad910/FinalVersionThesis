import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pyodbc

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

conn = pyodbc.connect("Driver={SQL Server Native Client 11.0};"
                      "Server=(localdb)\mssqllocaldb;"
                      "Database=METADATA2;"
                      "Trusted_Connection=yes;")

SQL_Query = pd.read_sql_query("SELECT * FROM MyView WHERE MetaLibraryKey = 40 AND StatusMainProcess = 'SUCCESS'", conn)
df = pd.DataFrame(SQL_Query)

df['DurationSecondsMainProcess'] = pd.to_numeric(df['DurationSecondsMainProcess'], errors='coerce')
# df['DurationSecondsSubProcess'] = pd.to_numeric(df['DurationSecondsSubProcess'], errors='coerce')
df['InsertSum'] = pd.to_numeric(df['InsertSum'], errors='raise')
df['SourceSum'] = pd.to_numeric(df['SourceSum'], errors='raise')
df['DeleteSum'] = pd.to_numeric(df['DeleteSum'], errors='raise')
df['UpdateSum'] = pd.to_numeric(df['UpdateSum'], errors='raise')
df['ExecutionDay'] = pd.to_numeric(df['ExecutionDay'], errors='raise')

df['StartMainProcess'] = pd.to_datetime(df['StartMainProcess'])
df['EndMainProcess'] = pd.to_datetime(df['EndMainProcess'])
# df['StartSubProcess'] = pd.to_datetime(df['StartSubProcess'])
# df['EndSubProcess'] = pd.to_datetime(df['EndSubProcess'])
# for 454 MS

#mask = (df['StartMainProcess'] > '2016-06-01') & (df['StartMainProcess'] < '2017-05-17')
#df = df.loc[mask]
#df.reset_index(drop=True, inplace=True)
temp = df['DurationSecondsMainProcess']
temp = np.array(temp)
print(len(temp))
dates = pd.date_range('2016-06-01', '2017-05-17')
print(df)

#Test
#mask = (df['StartMainProcess'] > '2018-09-01') & (df['StartMainProcess'] < '2019-03-30')
#df = df.loc[mask]
#print(df)
#df.reset_index(drop=True, inplace=True)

#For 721
#mask = (df['StartMainProcess'] > '2014-10-20') & (df['StartMainProcess'] < '2015-09-25')
#df = df.loc[mask]
#df.reset_index(drop=True, inplace=True)
#temp = df['DurationSecondsMainProcess']
#temp = np.array(temp)
#print(len(temp))
#dates = pd.date_range('2014-06-20', '2015-09-25')
#print(len(dates))



#mask = (df['StartMainProcess'] > '2017-01-01') & (df['StartMainProcess'] < '2018-02-01')
#df = df.loc[mask]

"""
test = pd.date_range('2016-02-01', '2017-12-01')
print(len(test))
test2 = pd.date_range('2017-12-01', '2018-12-28')
print(len(test2))
"""
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.05, 0.65, 0.5, 0.3])
ax1.plot(df['StartMainProcess'], df['DurationSecondsMainProcess'], color='blue')
ax1.set_title('Date vs DurationSeconds')
ax2.plot(df['StartMainProcess'], df['InsertSum'], color='green')
ax2.set_title('Date vs InsertSum')
plt.show()
