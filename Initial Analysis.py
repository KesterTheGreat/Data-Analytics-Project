
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import warnings
warnings.filterwarnings('ignore')

Lifeexp = pd.read_csv('E:\SCHOOL\CIND 820 PROJECT\Life Expectancy Data.csv', delimiter=',' ,dtype=None, index_col='Country')

Lifeexp.dataframeName= 'Life Expectancy Data.csv'

print(Lifeexp.head())
print(Lifeexp.describe())

Lifeexp.info()

#Lets get the percentage of missing values in each column or variable
PctMissingValues = Lifeexp.isnull().sum()*100/Lifeexp.isnull().count()
print(PctMissingValues)

#Lets get the actual number of missing values in each column
No_of_MissingValues = Lifeexp.isnull().sum()  
print(No_of_MissingValues)

#Obtaining the number of unique values
unique_values = Lifeexp.nunique()
print(unique_values)

#Utilizing interpolate to treat missing values
Lifeexp.dropna()
Lifeexp1 = Lifeexp.interpolate(method='linear', axis=0).ffill().bfill()
print(Lifeexp1)
#Make sure missing values have been treated and filled
Lifeexp1.isnull().sum()























