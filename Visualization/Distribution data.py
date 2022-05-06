import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


df = pd.read_csv(r'.\Normalized_April27.csv', encoding = "ISO-8859-1", engine='python')
# X = df.iloc[:, 1:73]

All_features=['gender','language','Age_Surg','Age_lab','years_between_surg_lab','REE','VCO2','VCO2kg','RQ','VECO2','curre_pa','paintens','D_sitting','we_regul','we_produ','suprtgrp','infomeet','smoker','gp6month','arthmeds','bpmeds','diabmeds','hartmeds','lipimeds','calcmeds','vitdmeds','other_sx','hospital','vomit12w','vomitnow','appetite','acid','asthma','clots','diabetes','heartdis','highbp','hilipids','jointpai','sleepapn','stomulce','incontin','stroke','thyroid',	'Android Region (%Fat)','Android Fat (g)','Android Total Mass (kg)','Gynoid Region (%Fat)','Gynoid Fat (g)','Gynoid Total Mass (kg)','Total Tissue (%Fat)','Total Region (%Fat)','Total Fat (g)','Total Lean (g)','Total Total Mass (kg)']

new_set=['vitdmeds','other_sx','hospital','vomit12w','vomitnow','appetite','acid','asthma','clots','diabetes','heartdis','highbp','hilipids','jointpai','sleepapn','stomulce','incontin','stroke','thyroid',	'Android Region (%Fat)','Android Fat (g)','Android Total Mass (kg)','Gynoid Region (%Fat)','Gynoid Fat (g)','Gynoid Total Mass (kg)','Total Tissue (%Fat)','Total Region (%Fat)','Total Fat (g)','Total Lean (g)','Total Total Mass (kg)']

X = df[new_set]
Y = df['Reversal_rate']
plt.hist(df["Android Fat (g)"], bins= 50)
plt.show()

for i, col in enumerate(df[new_set].columns):
    plt.hist(df[col], bins= 100)
    plt.title(df[col].name)
    plt.show()

#Show all dtypes
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#  print(df.dtypes)

