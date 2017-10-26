import pandas as pd
df=pd.read_csv('raw.csv')
y=df.get(['model', 'bought', 'month', 'issues'])
print y
