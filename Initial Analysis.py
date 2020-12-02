
import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import matplotlib.pyplot as plt
#pip install plotly
import plotly.express as px 
from plotly.offline import download_plotlyjs,  plot
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sqlalchemy import create_engine
import warnings
from scipy import stats
import seaborn as sns
#pip install missingno
import missingno as msno
import csv

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

Avg_mort = round(Lifeexp[['A_Mortality','Life_Exp']].mean(),2)

#Preliminary pruning of obvious outliers.

#Adult mortality rates lower than the 5th percentile
mortality_less_5_per = np.percentile(Lifeexp["A_Mortality"].dropna(),5) 
Lifeexp["A_Mortality"] = Lifeexp.apply(lambda x: np.nan if x["A_Mortality"] < mortality_less_5_per else x["A_Mortality"], axis=1)
#Remove Infant deaths of 0
Lifeexp["Infant_Deaths"] = Lifeexp["Infant_Deaths"].replace(0,np.nan)
#Remove the BMI less than 10 or greater than 50
Lifeexp["BMI"] =Lifeexp.apply(lambda x : np.nan if (x["BMI"] <10 or x["BMI"] >50) else x["BMI"],axis =1)
#Remove Under five deaths of 0
Lifeexp["Under_Five_Deaths"] =Lifeexp["Under_Five_Deaths"].replace(0,np.nan)


#MISSING VALUES

#Lets get the percentage of missing values in each column or variable
PctMissingValues = Lifeexp.isnull().sum()*100/Lifeexp.isnull().count()
print(PctMissingValues)

#Lets get the actual number of missing values in each column
No_of_MissingValues = Lifeexp.isnull().sum()  
print(No_of_MissingValues)


#visualization showing missing values across all variables
msno.matrix(Lifeexp)

#Since almost half of the BMI column is now null. 
#I've decided to drop this variable completely
Lifeexp.drop(columns='BMI', inplace=True)



#Obtaining the number of unique values
unique_values = Lifeexp.nunique()
print(unique_values)


#Utilizing imputation we will fill the missing values for each of the year's modes
Lifeexp1 = []

for year in list(Lifeexp.Year.unique()):
    year_data = Lifeexp[Lifeexp.Year == year].copy()
    
    for col in list(year_data.columns)[2:]:
        year_data[col] = year_data[col].fillna(year_data[col].dropna().mode()[0]).copy()

    Lifeexp1.append(year_data)
Lifeexp1 = pd.concat(Lifeexp1).copy()



#Make sure missing values have been treated and filled
No_of_MissingValues1 = Lifeexp1.isnull().sum()  
print(No_of_MissingValues1)




#Making a copy of the dataset which excludes 'Status' and 'year'
#This will be used for accurately counting the outliers and calculating the Z-scores
Lifeexp2 = Lifeexp1.drop(Lifeexp1.columns[[0,1]], axis=1)
print(Lifeexp2)


#OUTLIERS

#we'll use boxplots to show the distributions and detect outliers

#Creating a dictionary for the boxplot
lifeexp_dict = {'Life_Exp':1,'A_Mortality':2,
            'Infant_Deaths':3,'Alcohol':4,
            'Pct_Exp':5,'HepatitisB':6,'Measles':7,
            'Under_Five_Deaths':8,'Polio':9,
            'Total_Exp':10,'Diphtheria':11,'HIV/AIDS':12,
            'GDP':13,'Population':14,'thin_1-19':15,
            'thin_5-9':16,'Income_Comp_Of_Res':17,'Schooling':18}

# Detect outliers in each variable using box plots.
plt.figure(figsize=(20,30))

for variable,i in lifeexp_dict.items():
                     plt.subplot(5,7,i)
                     plt.boxplot(Lifeexp1[variable],whis=1.5)
                     plt.title(variable)
plt.show()           

#Histogram distributions
lifeexp_hist = ['Life_Exp','A_Mortality',
            'Infant_Deaths','Alcohol',
            'Pct_Exp','HepatitisB','Measles',
            'Under_Five_Deaths','Polio',
            'Total_Exp','Diphtheria','HIV/AIDS',
            'GDP','Population','thin_1-19',
            'thin_5-9','Income_Comp_Of_Res','Schooling']

plt.figure(figsize=(15,75))

for i in range(len(lifeexp_hist)):
    plt.subplot(18,5,i+1)
    plt.hist(Lifeexp1[lifeexp_hist[i]])
    plt.title(lifeexp_hist[i])

plt.show()

#--------------------------------------------------------------------

def no_of_outliers(col, data=Lifeexp1):
    
    print("\n"+15*'-' + col + 15*'-'+"\n")
    
    q75, q25 = np.percentile(data[col], [75, 25])
    iqr = q75 - q25
    min_val = q25 - (iqr*1.5)
    max_val = q75 + (iqr*1.5)
    
    no_of_outliers = len(np.where((data[col] > max_val) | (data[col] < min_val))[0])
    pct_of_outliers = round(no_of_outliers/len(data[col])*100, 2) 
    
    print('Number of outliers: {}'.format(no_of_outliers))
    print('Percent of data that is outlier: {}%'.format(pct_of_outliers))
    
outlier_list = list(Lifeexp2)
for col in outlier_list:
    no_of_outliers(col)


#Descriptive statistics
print(Lifeexp2.describe())



#using 'Lifeexp2' dataset, we will calculate the Z-scores, 
#the z-cores over 3 or under -3 are outliers
cols = list(Lifeexp2.columns)
Lifeexp2[cols]
for col in cols:
    col_zscore = col + '_zscore'
    Lifeexp2[col_zscore] = (Lifeexp2[col] - Lifeexp2[col].mean())/Lifeexp2[col].std(ddof=0)
    print(Lifeexp2)



#Utilizing the winsorization method we will replace outliers.

from scipy.stats.mstats import winsorize

winsor_life_exp = winsorize(Lifeexp1['Life_Exp'],(0.01,0))
winsor_a_mortality = winsorize(Lifeexp1['A_Mortality'],(0,0.04))
winsor_infant_deaths = winsorize(Lifeexp1['Infant_Deaths'],(0,0.13))
winsor_Alcohol = winsorize(Lifeexp1['Alcohol'],(0,0.0025))
winsor_pct_exp = winsorize(Lifeexp1['Pct_Exp'],(0,0.135))
winsor_HepatitisB = winsorize(Lifeexp1['HepatitisB'],(0.11,0))
winsor_Measles = winsorize(Lifeexp1['Measles'],(0,0.19))
winsor_Under_Five_Deaths = winsorize(Lifeexp1['Under_Five_Deaths'],(0,0.15))
winsor_Polio = winsorize(Lifeexp1['Polio'],(0.1,0))
winsor_total_exp = winsorize(Lifeexp1['Total_Exp'],(0,0.02))
winsor_Diphtheria = winsorize(Lifeexp1['Diphtheria'],(0.105,0))
winsor_HIV = winsorize(Lifeexp1['HIV/AIDS'],(0,0.185))
winsor_GDP = winsorize(Lifeexp1['GDP'],(0,0.15))
winsor_Population = winsorize(Lifeexp1['Population'],(0,0.15))
winsor_thinness_1to19yrs = winsorize(Lifeexp1['thin_1-19'],(0,0.035))
winsor_thinness_5to9yrs = winsorize(Lifeexp1['thin_5-9'],(0,0.035))
winsor_Income_Comp_Of_Res = winsorize(Lifeexp1['Income_Comp_Of_Res'],(0.10,0))
winsor_Schooling = winsorize(Lifeexp1['Schooling'],(0.025,0.01))

#Adding these new winsorized variables to the dataset

Lifeexp1['winsor_life_exp'] = winsor_life_exp
Lifeexp1['winsor_a_mortality'] = winsor_a_mortality
Lifeexp1['winsor_infant_deaths'] = winsor_infant_deaths
Lifeexp1['winsor_Alcohol'] = winsor_Alcohol
Lifeexp1['winsor_pct_exp'] = winsor_pct_exp
Lifeexp1['winsor_HepatitisB'] = winsor_HepatitisB
Lifeexp1['winsor_Measles'] = winsor_Measles
Lifeexp1['winsor_Under_Five_Deaths'] = winsor_Under_Five_Deaths
Lifeexp1['winsor_Polio'] = winsor_Polio
Lifeexp1['winsor_total_exp'] = winsor_total_exp
Lifeexp1['winsor_Diphtheria'] = winsor_Diphtheria
Lifeexp1['winsor_HIV'] = winsor_HIV
Lifeexp1['winsor_GDP'] = winsor_GDP
Lifeexp1['winsor_Population'] = winsor_Population
Lifeexp1['winsor_thinness_1to19yrs'] = winsor_thinness_1to19yrs
Lifeexp1['winsor_thinness_5to9yrs'] = winsor_thinness_5to9yrs
Lifeexp1['winsor_Income_Comp_Of_Res'] = winsor_Income_Comp_Of_Res
Lifeexp1['winsor_Schooling'] = winsor_Schooling

print(Lifeexp1)

#Second dictionary for detecting outliers after winsorization

lifeexp_dict2 = {'winsor_life_exp':1,'winsor_a_mortality':2,
            'winsor_infant_deaths':3,'winsor_Alcohol':4,
            'winsor_pct_exp':5,'winsor_HepatitisB':6,'winsor_Measles':7, 
            'winsor_Under_Five_Deaths':9,'winsor_Polio':10,
            'winsor_total_exp':11,'winsor_Diphtheria':12,'winsor_HIV':13,
            'winsor_GDP':14,'winsor_Population':15,'winsor_thinness_1to19yrs':16,
            'winsor_thinness_5to9yrs':17,'winsor_Income_Comp_Of_Res':18,'winsor_Schooling':19}

# Detect outliers in each variable using box plots.
plt.figure(figsize=(15,75))

for variable,i in lifeexp_dict2.items():
                     plt.subplot(16,6,i)
                     plt.boxplot(Lifeexp1[variable],whis=1.5)
                     plt.title(variable)
plt.show()  


desc = Lifeexp1.describe()



#Histogram distributions
lifeexp_hist2 = ['winsor_life_exp','winsor_a_mortality',
            'winsor_infant_deaths','winsor_Alcohol',
            'winsor_pct_exp','winsor_HepatitisB','winsor_Measles',  
            'winsor_Under_Five_Deaths','winsor_Polio',
            'winsor_total_exp','winsor_Diphtheria','winsor_HIV',
            'winsor_GDP','winsor_Population','winsor_thinness_1to19yrs',
            'winsor_thinness_5to9yrs','winsor_Income_Comp_Of_Res','winsor_Schooling']

plt.figure(figsize=(15,75))

for i in range(len(lifeexp_hist2)):
    plt.subplot(16,5,i+1)
    plt.hist(Lifeexp1[lifeexp_hist2[i]])
    plt.title(lifeexp_hist2[i])

plt.show()



# Descriptive statistics of categorical variables.
categ = Lifeexp1.describe(include=['O'])
print(categ)




#Creating a dataset which includes just the winsorized variables from the full dataset.
winsor_life_exp = Lifeexp1[['Year','Status','winsor_life_exp','winsor_a_mortality',
            'winsor_infant_deaths','winsor_Alcohol',
            'winsor_pct_exp','winsor_HepatitisB','winsor_Measles',  
            'winsor_Under_Five_Deaths','winsor_Polio',
            'winsor_total_exp','winsor_Diphtheria','winsor_HIV',
            'winsor_GDP','winsor_Population','winsor_thinness_1to19yrs',
            'winsor_thinness_5to9yrs','winsor_Income_Comp_Of_Res','winsor_Schooling']]

print(winsor_life_exp)


# Using the winsorized dataset we will take a broad look a the rise in Life Expectancy from 2000 - 2015 using bar plot.
plt.figure(figsize=(8,6))
plt.bar(winsor_life_exp.groupby('Year')['Year'].count().index,
        winsor_life_exp.groupby('Year')['winsor_life_exp'].mean(),
        color='purple',alpha=0.80)
plt.xlabel("Year",fontsize=10)
plt.ylabel("Average Life Expectancy",fontsize=10)
plt.title("Life Expectancy through the years")
plt.show()


# create another barplot to give us a general idea of how life expectancy levels between developed and developing countries.
plt.figure(figsize=(8,6))
plt.bar(winsor_life_exp.groupby('Status')['Status'].count().index,
        winsor_life_exp.groupby('Status')['winsor_life_exp'].mean(),
        color= 'green',alpha=0.80)
plt.xlabel("Status",fontsize=12)
plt.ylabel("Average Life Expectancy",fontsize=12)
plt.title("Life Expectancy by country status")
plt.show()


#creating a copy of our original dataset but reseting country index which lets us call that column much easily.

life_exp_visual = winsor_life_exp.copy(deep = True)
life_exp_visual.reset_index(level=0, inplace=True)
print(life_exp_visual)


#Utilizing the Plotly library to display 3 world maps, each displaying either the Life Expectancy, Income Composition of Resources, or No of Years of Schooling per country

#ISO list obtained from UN TRADE STATISTICS KNOWLEDGE
#Decided it was a better alternative to making 193 countries into a dictionary

country_map_list = {
'ABW':'Aruba', 'AFG':'Afghanistan','AGO':'Angola','AIA':'Anguilla','ALA':'Åland Islands','ALB':'Albania',
'AND':'Andorra','ARE':'United Arab Emirates','ARG':'Argentina','ARM':'Armenia','ASM':'American Samoa',
'ATA':'Antarctica','ATF':'French Southern Territories','ATG':'Antigua and Barbuda','AUS':'Australia',
'AUT':'Austria','AZE':'Azerbaijan','BDI':'Burundi','BEL':'Belgium','BEN':'Benin',
'BES':'Bonaire, Sint Eustatius and Saba','BFA':'Burkina Faso','BGD':'Bangladesh','BGR':'Bulgaria',
'BHR':'Bahrain','BHS':'Bahamas','BIH':'Bosnia and Herzegovina','BLM':'Saint Barthélemy','BLR':'Belarus',
'BLZ':'Belize','BMU':'Bermuda','BOL':'Bolivia (Plurinational State of)','BRA':'Brazil','BRB':'Barbados',
'BRN':'Brunei Darussalam','BTN':'Bhutan','BVT':'Bouvet Island','BWA':'Botswana','CAF':'Central African Republic',
'CAN':'Canada','CCK':'Cocos (Keeling) Islands','CHE':'Switzerland','CHL':'Chile','CHN':'China',
'CIV':'Côte d\'Ivoire','CMR':'Cameroon','COD':'Democratic Republic of the Congo','COG':'Congo',
'COK':'Cook Islands','COL':'Colombia','COM':'Comoros','CPV':'Cabo Verde','CRI':'Costa Rica','CUB':'Cuba',
'CUW':'Curaçao','CXR':'Christmas Island','CYM':'Cayman Islands','CYP':'Cyprus','CZE':'Czechia',
'DEU':'Germany','DJI':'Djibouti','DMA':'Dominica','DNK':'Denmark','DOM':'Dominican Republic','DZA':'Algeria',
'ECU':'Ecuador','EGY':'Egypt','ERI':'Eritrea','ESH':'Western Sahara','ESP':'Spain','EST':'Estonia',
'ETH':'Ethiopia','FIN':'Finland','FJI':'Fiji','FLK':'Falkland Islands (Malvinas)','FRA':'France',
'FRO':'Faroe Islands','FSM':'Micronesia (Federated States of)','GAB':'Gabon',
'GBR':'United Kingdom of Great Britain and Northern Ireland','GEO':'Georgia','GGY':'Guernsey','GHA':'Ghana',
'GIB':'Gibraltar','GIN':'Guinea','GLP':'Guadeloupe','GMB':'Gambia','GNB':'Guinea-Bissau',
'GNQ':'Equatorial Guinea','GRC':'Greece','GRD':'Grenada','GRL':'Greenland','GTM':'Guatemala',
'GUF':'French Guiana','GUM':'Guam','GUY':'Guyana','HKG':'Hong Kong','HMD':'Heard Island and McDonald Islands',
'HND':'Honduras','HRV':'Croatia','HTI':'Haiti','HUN':'Hungary','IDN':'Indonesia','IMN':'Isle of Man',
'IND':'India','IOT':'British Indian Ocean Territory','IRL':'Ireland','IRN':'Iran (Islamic Republic of)',
'IRQ':'Iraq','ISL':'Iceland','ISR':'Israel','ITA':'Italy','JAM':'Jamaica','JEY':'Jersey','JOR':'Jordan',
'JPN':'Japan','KAZ':'Kazakhstan','KEN':'Kenya','KGZ':'Kyrgyzstan','KHM':'Cambodia','KIR':'Kiribati',
'KNA':'Saint Kitts and Nevis','KOR':'Republic of Korea','KWT':'Kuwait','LAO':'Lao People\'s Democratic Republic',
'LBN':'Lebanon','LBR':'Liberia','LBY':'Libya','LCA':'Saint Lucia', 'LIE':'Liechtenstein','LKA':'Sri Lanka',
'LSO':'Lesotho','LTU':'Lithuania','LUX':'Luxembourg','LVA':'Latvia','MAC':'Macao',
'MAF':'Saint Martin (French part)','MAR':'Morocco','MCO':'Monaco','MDA':'Republic of Moldova',
'MDG':'Madagascar','MDV':'Maldives','MEX':'Mexico','MHL':'Marshall Islands','MKD':'North Macedonia',
'MLI':'Mali','MLT':'Malta','MMR':'Myanmar','MNE':'Montenegro','MNG':'Mongolia','MNP':'Northern Mariana Islands',
'MOZ':'Mozambique','MRT':'Mauritania','MSR':'Montserrat','MTQ':'Martinique','MUS':'Mauritius','MWI':'Malawi',
'MYS':'Malaysia','MYT':'Mayotte','NAM':'Namibia','NCL':'New Caledonia','NER':'Niger','NFK':'Norfolk Island',
'NGA':'Nigeria','NIC':'Nicaragua','NIU':'Niue','NLD':'Netherlands','NOR':'Norway','NPL':'Nepal','NRU':'Nauru',
'NZL':'New Zealand','OMN':'Oman','PAK':'Pakistan','PAN':'Panama','PCN':'Pitcairn','PER':'Peru','PHL':'Philippines',
'PLW':'Palau','PNG':'Papua New Guinea','POL':'Poland','PRI':'Puerto Rico','PRK':'Democratic People\'s Republic of Korea',
'PRT':'Portugal','PRY':'Paraguay','PSE':'Palestine, State of','PYF':'French Polynesia','QAT':'Qatar','REU':'Réunion',
'ROU':'Romania','RUS':'Russian Federation','RWA':'Rwanda','SAU':'Saudi Arabia','SDN':'Sudan','SEN':'Senegal',
'SGP':'Singapore','SGS':'South Georgia and the South Sandwich Islands','SHN':'Saint Helena, Ascension and Tristan da Cunha',
'SJM':'Svalbard and Jan Mayen','SLB':'Solomon Islands','SLE':'Sierra Leone','SLV':'El Salvador','SMR':'San Marino',
'SOM':'Somalia','SPM':'Saint Pierre and Miquelon','SRB':'Serbia','SSD':'South Sudan','STP':'Sao Tome and Principe',
'SUR':'Suriname','SVK':'Slovakia','SVN':'Slovenia','SWE':'Sweden','SWZ':'Eswatini','SXM':'Sint Maarten (Dutch part)',
'SYC':'Seychelles','SYR':'Syrian Arab Republic','TCA':'Turks and Caicos Islands','TCD':'Chad','TGO':'Togo',
'THA':'Thailand','TJK':'Tajikistan','TKL':'Tokelau','TKM':'Turkmenistan','TLS':'Timor-Leste','TON':'Tonga',
'TTO':'Trinidad and Tobago','TUN':'Tunisia','TUR':'Turkey','TUV':'Tuvalu','TWN':'Taiwan, Province of China',
'TZA':'United Republic of Tanzania','UGA':'Uganda','UKR':'Ukraine','UMI':'United States Minor Outlying Islands',
'URY':'Uruguay','USA':'United States of America','UZB':'Uzbekistan','VAT':'Holy See','VCT':'Saint Vincent and the Grenadines',
'VEN':'Venezuela (Bolivarian Republic of)','VGB':'Virgin Islands (British)','VIR':'Virgin Islands (U.S.)','VNM':'Viet Nam',
'VUT':'Vanuatu','WLF':'Wallis and Futuna','WSM':'Samoa','YEM':'Yemen','ZAF':'South Africa','ZMB':'Zambia',
'ZWE':'Zimbabwe'}

# Turning the above dict into dataframe

country_map_df = pd.DataFrame.from_dict(country_map_list, orient='index', columns=['Country'])

# I grouped the new dataframe by Country and combined the variables we want to plot with their respective mathematical classifications


country_map_agg = life_exp_visual.groupby('Country')['winsor_life_exp', 'Status', 
                                               'winsor_Income_Comp_Of_Res', 'winsor_Schooling'].agg(
    {'winsor_life_exp':'mean', 'Status':'min', 'winsor_Income_Comp_Of_Res':'mean', 'winsor_Schooling':'mean'}) 
                                                   
# Merging country_map_df and country_map_agg together

merged_country_map = pd.merge(country_map_agg, country_map_df, left_index=True, right_on='Country')

#Utilizing the Plotly library to display 3 world maps
W_map = make_subplots(
    rows=3, cols=1,
    row_heights=[0.25, 0.25, 0.25],
    vertical_spacing=0.015,
    subplot_titles=("Life Expectancy by Country", "Income Composition of Resources", "Average No of Years of Schooling per country"),
    specs=[[{"type": "Choropleth", "rowspan": 1}], [{"type": "Choropleth", "rowspan": 1}], [{"type": "Choropleth", "rowspan": 1}]])

# Life Expectancy
W_map.add_trace(go.Choropleth(locations = merged_country_map.index,
                  z= merged_country_map['winsor_life_exp'], 
                  text=merged_country_map['Country'],
                  name='Life Expectancy',
                  colorbar={'title':'Life<br>Expectancy', 'len':.25, 'x':.99,'y':.850},
                  colorscale='inferno'), row=1,col=1)

# Income Composition of Resources
W_map.add_trace(go.Choropleth(locations = merged_country_map.index,
                  z= merged_country_map['winsor_Income_Comp_Of_Res'], 
                  text=merged_country_map['Country'],
                  name='Income Composition of Resources',
                  colorbar={'title':'Resources<br>Index', 'len':.24, 'x':.99,'y':.505},
                  colorscale='Portland'), row=2,col=1)

#Average No of Years of Schooling per country
W_map.add_trace(go.Choropleth(locations = merged_country_map.index,
                  z= merged_country_map['winsor_Schooling'], 
                  text=merged_country_map['Country'],
                  name='Average No of Years of Schooling per country',
                  colorbar={'title':'Years of<br>Schooling', 'len':.248, 'x':.99,'y':.169},
                  colorscale='magma'), row=3,col=1)

W_map.update_layout(margin=dict(r=1, t=30, b=10, l=30),
                    width=700,
                    height=1400)

plot(W_map)

W_map.write_html("E:\SCHOOL\CIND 820 PROJECT\Themap.html")

#Created a Pie Chart displaying the TOP 20 countries with the highest average Life Expectancy
#This gives us an idea of the countries doing the best.

Life_exp_pie = life_exp_visual.groupby(life_exp_visual['Country'])['winsor_life_exp'].mean()
Life_exp_pie = pd.DataFrame(index = life_exp_visual["Country"].unique() , data = Life_exp_pie)
Life_exp_pie = Life_exp_pie.sort_values(by = "winsor_life_exp" , ascending = False)
Life_exp_pie = Life_exp_pie.head(20)
 
The_pie = px.pie(data_frame = Life_exp_pie , 
            names = Life_exp_pie.index , 
            values = "winsor_life_exp" , 
            template = "seaborn" , 
             opacity = 1.0 , 
            color_discrete_sequence=px.colors.sequential.Plasma , 
            hole = 0.5)

The_pie.update_traces (pull= 0.05 , textinfo = "percent+label" , rotation = 90)

The_pie.update_layout(
    title = "Pie Chart" , 
    paper_bgcolor = 'rgb(230, 230 , 230)' , 
    plot_bgcolor = 'rgb(243, 243 , 243)',
    annotations=[dict(text='Top 20 Countries with the highest<br>means of life expectancy', font_size=17, showarrow=False)]
)

plot(The_pie)





# Line plots showing the behaviours of Life Expectancy and Adult Mortality over the years

sns.lineplot('Year', 'winsor_life_exp', data=winsor_life_exp, marker='o')
plt.title('Life Expectancy by Year')
plt.xlabel("Year",fontsize=12)
plt.ylabel("Average Life Expectancy",fontsize=12)
plt.show()

sns.lineplot('Year', 'winsor_a_mortality', data=winsor_life_exp, marker='o')
plt.title('Adult Mortality by Year')
plt.xlabel("Year",fontsize=12)
plt.ylabel("Adult Mortality",fontsize=12)
plt.show()


#Scatter plots showing the relationship between Life expectancy and all continuous variables in our winsorized dataset.

plt.figure(figsize=(18,40))

plt.subplot(6,3,1)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_a_mortality"])
plt.title("Life Expectancy vs Adult Mortality")

plt.subplot(6,3,2)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_infant_deaths"])
plt.title("Life Expectancy vs Infant Deaths")

plt.subplot(6,3,3)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_Alcohol"])
plt.title("Life Expectancy vs Alcohol")

plt.subplot(6,3,4)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_pct_exp"])
plt.title("Life Expectancy vs Percentage Exp")

plt.subplot(6,3,5)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_HepatitisB"])
plt.title("Life Expectancy vs HepatitisB")

plt.subplot(6,3,6)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_Measles"])
plt.title("Life Expectancy vs Measles")

plt.subplot(6,3,7)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_Under_Five_Deaths"])
plt.title("Life Expectancy vs Under Five Deaths")

plt.subplot(6,3,8)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_Polio"])
plt.title("Life Expectancy vs Polio")

plt.subplot(6,3,9)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_total_exp"])
plt.title("Life Expectancy vs Total Expenditure")

plt.subplot(6,3,10)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_Diphtheria"])
plt.title("Life Expectancy vs Diphtheria")

plt.subplot(6,3,11)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_HIV"])
plt.title("Life Expectancy vs HIV")

plt.subplot(6,3,12)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_GDP"])
plt.title("Life Expectancy vs GDP")

plt.subplot(6,3,13)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_Population"])
plt.title("Life Expectancy vs Population")

plt.subplot(6,3,14)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_thinness_1to19yrs"])
plt.title("Life Expectancy vs thinness 1to19 years")

plt.subplot(6,3,15)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_thinness_5to9yrs"])
plt.title("Life Expectancy vs thinness 5to9 years")

plt.subplot(6,3,16)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_Income_Comp_Of_Res"])
plt.title("Life Expectancy vs Income Comp Of Resources")

plt.subplot(6,3,17)
plt.scatter(winsor_life_exp["winsor_life_exp"], winsor_life_exp["winsor_Schooling"])
plt.title("Life Expectancy vs Schooling")


plt.show()



#Let us review a scatter plot showing the relationship between Income composition of resources and schooling.
plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.scatter(winsor_life_exp["winsor_Schooling"], winsor_life_exp["winsor_Income_Comp_Of_Res"])
plt.title("Schooling vs Income Comp Of Resources")
plt.show()

    
    
    
#FIRST RESEARCH QUESTION
#created a dataset which excludes the year, just for the heatmap
heat_life_exp = winsor_life_exp[['winsor_life_exp','winsor_a_mortality',
            'winsor_infant_deaths','winsor_Alcohol',
            'winsor_pct_exp','winsor_HepatitisB','winsor_Measles',  
            'winsor_Under_Five_Deaths','winsor_Polio',
            'winsor_total_exp','winsor_Diphtheria','winsor_HIV',
            'winsor_GDP','winsor_Population','winsor_thinness_1to19yrs',
            'winsor_thinness_5to9yrs','winsor_Income_Comp_Of_Res','winsor_Schooling']]

#created a function for heatmap of our dataset.
def life_exp_heatmap():
    plt.figure(figsize=(20,15))
    sns.heatmap(heat_life_exp.corr(), annot=True, fmt='.2g', vmin=-1, vmax=1, center=0, cmap='YlGnBu')
    plt.ylim(len(heat_life_exp.columns), 0)
    plt.title('Life Expectancy Correlation Matrix')
    plt.show()
life_exp_heatmap()


#SECOND RESEARCH QUESTION
round(Lifeexp1[['Status','Life_Exp']].groupby(['Status']).mean(),2)
long = Lifeexp1.loc[Lifeexp1['Status']=='Developed','Life_Exp']
short = Lifeexp1.loc[Lifeexp1['Status']=='Developing','Life_Exp']
stat = stats.ttest_ind(long, short)
print(stat)



from sklearn.model_selection import GroupShuffleSplit 
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error


# PREDICTIVE MODELING


#Creating a copy of the dataset which resets the index. This dataset will be used for the modeling 
life_exp_train = Lifeexp1.copy(deep = True)
life_exp_train.reset_index(level=0, inplace=True)
life_exp_train.sort_values(by=['Year'], ascending=True, inplace=True, ignore_index=False)
print(life_exp_train)

independent = life_exp_train[['Year','winsor_a_mortality',
            'winsor_infant_deaths','winsor_Alcohol',
            'winsor_pct_exp','winsor_HepatitisB','winsor_Measles',  
            'winsor_Under_Five_Deaths','winsor_Polio',
            'winsor_total_exp','winsor_Diphtheria','winsor_HIV',
            'winsor_GDP','winsor_Population','winsor_thinness_1to19yrs',
            'winsor_thinness_5to9yrs','winsor_Income_Comp_Of_Res','winsor_Schooling']]
 
dependent = life_exp_train[['winsor_life_exp']]

#Splitting the dataset into a training and test datasets based on year
# TRAIN = 2000 - 2013
# TEST = 2014 - 2015

gs = GroupShuffleSplit(n_splits=2, train_size=.9, random_state=42)

train_ix, test_ix = next(gs.split(independent, dependent, groups = independent.Year))

X_train = independent.loc[train_ix]
y_train = dependent.loc[train_ix]
X_test = independent.loc[test_ix]
y_test = dependent.loc[test_ix]

y_train = np.reshape(y_train , (2572 , ))
y_test = np.reshape(y_test , (366 , ))
X_train.shape , X_test.shape , y_train.shape , y_test.shape


#Creating a dataset with only the most correlated attributes - Trimmed Dataset
Final_X = life_exp_train[['Year','winsor_a_mortality',
            'winsor_HIV','winsor_Income_Comp_Of_Res','winsor_Schooling']]
 
Final_Y = life_exp_train[['winsor_life_exp']]

#Splitting the dataset into a training and test datasets based on year
# TRAIN = 2000 - 2013
# TEST = 2014 - 2015

gs1 = GroupShuffleSplit(n_splits=2, train_size=.9, random_state=42)
train_ix, test_ix = next(gs1.split(Final_X, Final_Y, groups = Final_X.Year))
X_trainF = Final_X.loc[train_ix]
y_trainF = Final_Y.loc[train_ix]
X_testF = Final_X.loc[test_ix]
y_testF = Final_Y.loc[test_ix]


# ---------------------------------------------------

# LINEAR REGRESSION

# Import LASSO LINEAR REGRESSION MODEL

from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

#Full Dataset Linear Regression
lr = Lasso(alpha = 0.001 , max_iter = 5000)
lr.fit(X_train , y_train)
lr_predict = lr.predict(X_test)
print('LASSO FULL Dataset MAE = ',mean_absolute_error(y_test,lr_predict))
print('LASSO FULL Dataset RMSE = ',np.sqrt(mean_squared_error(y_test,lr_predict))) 
print('LASSO FULL Dataset RSquared Score = ',r2_score(y_test, lr_predict))



#Trimmed Dataset Linear Regression
lr1 = Lasso(alpha = 0.001 , max_iter = 5000)
lr1.fit(X_trainF , y_trainF)
lr_predict1 = lr1.predict(X_testF)
print('LASSO TRIMMED Dataset MAE = ',mean_absolute_error(y_testF,lr_predict1))
print('LASSO TRIMMED Dataset RMSE = ',np.sqrt(mean_squared_error(y_testF,lr_predict1))) 
print('LASSO TRIMMED Dataset RSquared Score = ',r2_score(y_testF, lr_predict1))






from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# NON LINEAR REGRESSION - FULL DATASET

# DECISION TREE REGRESSION MODEL

Life_tree = DecisionTreeRegressor(max_depth=8, min_samples_leaf=0.13, random_state=3)
Life_tree.fit(X_train, y_train)

# Print MAE, RMSE and R-squared value for regression tree 'Life_tree' on testing data
pred_test_tree = Life_tree.predict(X_test)
print('Decision Tree FULL Dataset MAE (weak) = ',mean_absolute_error(y_test,pred_test_tree))
print('Decision Tree FULL Dataset RMSE (weak) = ',np.sqrt(mean_squared_error(y_test,pred_test_tree))) 
print('Decision Tree FULL Dataset RSquared Score (weak) = ', r2_score(y_test, pred_test_tree))


#We will change the value of the parameter 'max_depth' to see how that affects the model performance.

#Changed Parameters (Max Depth)
Life_tree1 = DecisionTreeRegressor(max_depth=5)
Life_tree1.fit(X_train, y_train)


# Print MAE, RMSE and R-squared value for regression tree 'Life_tree1' on testing data
Test_pred = Life_tree1.predict(X_test)
print('Decision Tree FULL Dataset MAE = ',mean_absolute_error(y_test,Test_pred))
print('Decision Tree FULL Dataset RMSE = ',np.sqrt(mean_squared_error(y_test,Test_pred))) 
print('Decision Tree FULL Dataset RSquared Score = ',r2_score(y_test, Test_pred)) 




# RANDOM FOREST REGRESSION MODEL

model_rf = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
model_rf.fit(X_train, y_train) 
pred_test_rf = model_rf.predict(X_test)
print('Random Forest FULL Dataset MAE = ',mean_absolute_error(y_test,pred_test_rf))
print('Random Forest FULL Dataset RMSE = ',np.sqrt(mean_squared_error(y_test,pred_test_rf)))
print('Random Forest FULL Dataset RSquared Score = ',r2_score(y_test, pred_test_rf))
#plt.scatter(y_test, pred_test_rf)


# -------------------------------------------------------------------------


# NON LINEAR REGRESSION - TRIMMED DATASET

# DECISION TREE REGRESSION MODEL

Life_tree2 = DecisionTreeRegressor(max_depth=5)
Life_tree2.fit(X_trainF, y_trainF)

# Print MAE, RMSE and R-squared value for regression tree 'Life_tree2' on testing data
Test_pred1 = Life_tree2.predict(X_testF)
print('Decision Tree Trimmed Dataset MAE = ',mean_absolute_error(y_testF,Test_pred1))
print('Decision Tree Trimmed Dataset RMSE = ',np.sqrt(mean_squared_error(y_testF,Test_pred1))) 
print('Decision Tree Trimmed Dataset RSquared Score = ',r2_score(y_testF, Test_pred1)) 




# RANDOM FOREST REGRESSION MODEL

model_rf1 = RandomForestRegressor(n_estimators=500, oob_score=True, random_state=100)
model_rf1.fit(X_trainF, y_trainF) 
pred_test_rf1 = model_rf1.predict(X_testF)
print('Random Forest Trimmed Dataset MAE = ',mean_absolute_error(y_testF,pred_test_rf1))
print('Random Forest Trimmed Dataset RMSE = ',np.sqrt(mean_squared_error(y_testF,pred_test_rf1)))
print('Random Forest Trimmed Dataset RSquared Score = ',r2_score(y_testF, pred_test_rf1))

 


# We observed that the Random Forest model outperforms 
# the Regression Tree models, with the test set RMSE and R-squared values
# of 2.3 thousand and 92.3 percent, respectively. This is close to 
# the most ideal result of an R-squared value of 1, indicating the superior 
# performance of the Random Forest algorithm.






























































































