import pandas as pd
import statsmodels.api as sm

df = sm.datasets.fair.load_pandas().data
print(df.head())

df.to_csv(r"C:\Users\ankan.d\Desktop\Log_Reg\data_given\data.csv", index = False)

print("DONE")