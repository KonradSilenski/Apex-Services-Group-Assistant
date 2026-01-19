import pandas as pd

gapminder_csv_url = r"sorCodeModel/dataSOR/processedSORCodes2.csv"

record = pd.read_csv(gapminder_csv_url)
record.head()

print(record['Job Type'].unique())