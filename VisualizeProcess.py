import matplotlib.pyplot as plt
import pandas as pd
import pyodbc

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Alter the metaLibraryKey [39, 40, 454, 743] to visualize process
metaLibraryKey = '40'

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
df = pd.DataFrame(SQL_Query)

df['DurationSecondsMainProcess'] = pd.to_numeric(df['DurationSecondsMainProcess'], errors='coerce')
df['InsertSum'] = pd.to_numeric(df['InsertSum'], errors='raise')
df['SourceSum'] = pd.to_numeric(df['SourceSum'], errors='raise')
df['DeleteSum'] = pd.to_numeric(df['DeleteSum'], errors='raise')
df['UpdateSum'] = pd.to_numeric(df['UpdateSum'], errors='raise')
df['ExecutionDay'] = pd.to_numeric(df['ExecutionDay'], errors='raise')

df['StartMainProcess'] = pd.to_datetime(df['StartMainProcess'])
df['EndMainProcess'] = pd.to_datetime(df['EndMainProcess'])

dates = pd.date_range('2016-06-01', '2017-05-17')
print(df)

fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_axes([0, 0, 1, 1])
ax2 = fig.add_axes([0.05, 0.65, 0.5, 0.3])
ax1.plot(df['StartMainProcess'], df['DurationSecondsMainProcess'], color='blue')
ax1.set_title('Date vs DurationSeconds')
ax2.plot(df['StartMainProcess'], df['InsertSum'], color='green')
ax2.set_title('Date vs InsertSum')
plt.show()
