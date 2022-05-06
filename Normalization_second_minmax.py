import pandas as pd

df = pd.read_csv(r'April27_before_normalization.csv', encoding = "ISO-8859-1", engine='python')

normalized_df=(df-df.min())/(df.max()-df.min())

normalized_df.to_csv('Normalized_April27.csv')