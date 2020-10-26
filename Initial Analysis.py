
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import matplotlib.pyplot as plt
from sqlalchemy import create_engine
import warnings
from scipy import stats
warnings.filterwarnings('ignore')

Lifeexp = pd.read_csv('E:\SCHOOL\CIND 820 PROJECT\Life Expectancy Data.csv', delimiter=',' ,dtype=None, index_col='Country')

Lifeexp.dataframeName= 'Life Expectancy Data.csv'

print(Lifeexp.head())
Lifeexp.rename(columns={"Life expectancy ":"Life_Exp",
                        "Adult Mortality":"A_Mortality",             
                        "infant deaths":"Infant_Deaths",
                        "percentage expenditure":"Pct_Exp",
                        "Hepatitis B":"HepatitisB",
                        "Measles ":"Measles",
                        " BMI ":"BMI","under-five deaths ":
                        "Under_Five_Deaths","Diphtheria ":"Diphtheria",
                        " HIV/AIDS":"HIV/AIDS",
                        " thinness  1-19 years":"thin_1-19",
                        " thinness 5-9 years":"thin_5-9",
                        "Income composition of resources":"Income_Comp_Of_Res",
                        "Total expenditure":"Total_Exp"},inplace=True)

# Get the data types
print(Lifeexp.dtypes)
# or to get more explanatory information
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


#Making a dataset which excludes 'Status' and 'year'
Lifeexp2 = Lifeexp1.drop(Lifeexp1.columns[[0,1]], axis=1)
print(Lifeexp2)

#Making a copy of the above dataset
Lifeexpz = Lifeexp1.drop(Lifeexp1.columns[[0,1]], axis=1)
print(Lifeexpz)


#First we can use boxplots to show the distributions and detect outliers

# To create seperate boxplots for each column we need to
# create a dictionary of columns.
lifeexp_dict = {'Life_Exp':1,'A_Mortality':2,
            'Infant_Deaths':3,'Alcohol':4,
            'Pct_Exp':5,'HepatitisB':6,'Measles':7,
            'BMI':8,'Under_Five_Deaths':9,'Polio':10,
            'Total_Exp':11,'Diphtheria':12,'HIV/AIDS':13,
            'GDP':14,'Population':15,'thin_1-19':16,
            'thin_5-9':17,'Income_Comp_Of_Res':18,'Schooling':19}

# Detect outliers in each variable using box plots.
plt.figure(figsize=(20,30))

for variable,i in lifeexp_dict.items():
                     plt.subplot(5,7,i)
                     plt.boxplot(Lifeexp2[variable],whis=1.5)
                     plt.title(variable)
plt.show()           


#Descriptive statistics
print(Lifeexp2.describe())



#using 'Lifeexp2' dataset, i will calculate the Z-scores, 
#the z-cores over 3 or under -3 are outliers
cols = list(Lifeexp2.columns)
Lifeexp2[cols]
for col in cols:
    col_zscore = col + '_zscore'
    Lifeexp2[col_zscore] = (Lifeexp2[col] - Lifeexp2[col].mean())/Lifeexp2[col].std(ddof=0)
    print(Lifeexp2)



#for each row the z score is computed, 
#I will remove only those rows that 
#have z score greater than 3 or less than -3. 
Lifeexp3 = Lifeexpz[(Lifeexpz.apply(stats.zscore)>-3) & (Lifeexpz.apply(stats.zscore)<3)]
print(Lifeexp3)



#Utilizing interpolate again to treat removed outliers
Lifeexp3.dropna()
Lifeexp3 = Lifeexp.interpolate(method='linear', axis=0).ffill().bfill()
print(Lifeexp3)
#Make sure missing values have been treated and filled
Lifeexp3.isnull().sum()




















