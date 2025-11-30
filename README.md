# EDA-Loan-Case-Study
Performed detailed EDA on a loan dataset to identify variables contributing to loan default. Analyzed borrower demographics, credit history, income, debt-to-income ratio, and repayment behaviour. Generated insights that could guide risk segmentation and loan approval policies.
## Importing the Libraries.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 200)
pd.set_option('display.width', 1500)
pd.set_option('display.expand_frame_repr', False)

## Description and column details

Description = pd.read_csv("columns_description.csv", encoding='latin-1')
Description

## Reading the application data and understanding the data.

Current_Application_data= pd.read_csv("application_data.csv")

Current_Application_data.head()

# Number of rows and coloumns , We have 122 columns
Current_Application_data.shape 

#Data type of each coloumns , we have 65:float64, 41:int64 and 16: object
Current_Application_data.info(all) 

Current_Application_data.describe()

### Handaling Null columns in application data
_Get the null value percentage for each coloumns respectivily and decide whcich to drop_.

# Percentage of  missing values column wise
Current_Application_data.isnull().mean().round(4)*100

# Delete the columns having more than 40% missing values
drop_acoloumns=Current_Application_data.columns[Current_Application_data.isnull().mean().round(4)*100 > 40] 
drop_acoloumns

len(drop_acoloumns)

____We could see that there are 49 coloumns that have more that 40% of its value null , hence we can completely drop the coloumns.____

Current_Application_data.drop(drop_acoloumns,axis=1,inplace=True)

#confirmed that the 49 coumns have been deleted from from data 
Current_Application_data.shape 

missing_percentage = Current_Application_data.isnull().mean().round(4) * 100
missing = missing_percentage[(missing_percentage > 0) & (missing_percentage < 40)]
missing

# Cheking Outliers and identifying best way to replace missing values 

#### Checking the Missing Values and outliers in AMT_REQ_cols

#Define the columns
AMT_REQ_cols = ['AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK','AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR']

Current_Application_data[AMT_REQ_cols].describe()

### Find the outliers by BAR PLOT


fig, axes = plt.subplots(3, 2, figsize=(14, 8))
colors = sns.color_palette("Set2")

legend_labels = [f'{col} Values' for col in AMT_REQ_cols]
axes = axes.flatten()
for i, col in enumerate(AMT_REQ_cols):
    value_counts = Current_Application_data[col].value_counts()
    sns.barplot(x=value_counts.index, y=value_counts.values, ax=axes[i], palette=colors)
    axes[i].set_title(f"Bar Plot for {col}")
plt.tight_layout()
plt.show()


missing_value_counts = Current_Application_data[AMT_REQ_cols].isnull().sum()
print("Missing Value Counts:")
print(missing_value_counts)
print("_______________________________________________________________________________")
for col in AMT_REQ_cols:
    unique_values = Current_Application_data[col].unique()
    print(f"Unique values for {col}:")
    print(unique_values)
    print("_______________________________________________________________________________")

## Obserbation:  From the above visualisation we could identify its better to use Mode to impute the missing values

#### Checking the Missing Values and outliers in EXT_DF

EXT_Cols = ['EXT_SOURCE_2', 'EXT_SOURCE_3']
EXT_DataFrame = Current_Application_data[['EXT_SOURCE_2', 'EXT_SOURCE_3']]

Current_Application_data[EXT_Cols].describe()

### Find the outliers by BOX PLOT

sns.boxplot(data=EXT_DataFrame)
plt.title('Box Plot of EXT_SOURCE_2 and EXT_SOURCE_3')
plt.show()

missing_value_counts = Current_Application_data[EXT_Cols].isnull().sum()
print("Missing Value Counts:")
print(missing_value_counts)
print("_______________________________________________________________________________")


## Obserbation: So scattering in data identified its better to keep the data and for null we can use mean imputation

#### Checking the Missing Values and outliers in OBS_DFE_cols

OBS_DFE_cols = ['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']
OBS_DFE_DataFrame = Current_Application_data[['OBS_30_CNT_SOCIAL_CIRCLE','DEF_30_CNT_SOCIAL_CIRCLE','OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE']]

Current_Application_data[OBS_DFE_cols].describe()

### Find the outliers by SCATTER PLOT

plt.figure(figsize=(18, 6))
cmap = plt.get_cmap('viridis')
for col in OBS_DFE_cols:
    colors = np.arange(len(Current_Application_data))

    plt.scatter(Current_Application_data[col], Current_Application_data.index, c=colors, cmap=cmap)
    plt.title(f'Colorful Scatter Plot of {col}')
    plt.xlabel(col)
    plt.ylabel('Index')
    plt.colorbar(label='Color Map')
    plt.show()

missing_value_counts = Current_Application_data[OBS_DFE_cols].isnull().sum()
print("Missing Value Counts:")
print(missing_value_counts)
print("_______________________________________________________________________________")
for col in OBS_DFE_cols:
    unique_values = Current_Application_data[col].unique()
    print(f"Unique values for {col}:")
    print(unique_values)
    print("_______________________________________________________________________________")

### Obserbation : 
#### OBS_30_CNT_SOCIAL_CIRCLE,DEF_30_CNT_SOCIAL_CIRCLE,OBS_60_CNT_SOCIAL_CIRCLE,DEF_60_CNT_SOCIAL_CIRCLE  have Outliersso we cannot use mean mostly we can use Median

#### Checking the Missing Values and outliers in OCCUPATION_TYPE

Current_Application_data["OCCUPATION_TYPE"].describe()

### Find the outliers by BAR PLOT

occupation_data = Current_Application_data["OCCUPATION_TYPE"].value_counts()
total_occupations = len(Current_Application_data)
percentage_data = (occupation_data / total_occupations) * 100

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=occupation_data.index, y=occupation_data.values, palette="Set2")
for p, percentage in zip(ax.patches, percentage_data):
    ax.annotate(f"{percentage:.2f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center')
plt.xlabel("Occupation Type")
plt.ylabel("Count")
plt.title("Distribution of Occupation Types")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

missing_value_counts = Current_Application_data["OCCUPATION_TYPE"].isnull().sum()
print("Missing Value Counts:")
print(missing_value_counts)
print("_______________________________________________________________________________")
unique_values = Current_Application_data["OCCUPATION_TYPE"].unique()
print("Unique values for OCCUPATION_TYPE:")
print(unique_values)
print("_______________________________________________________________________________")


### Obserbation: The applicants are more from the OCCUPATION_TYPE Laborers

#### Checking the Missing Values and outliers in NAME_TYPE_SUITE 

Current_Application_data["NAME_TYPE_SUITE"].describe()

### Find the outliers by BAR PLOT

NAME_TYPE_SUITE_data = Current_Application_data["NAME_TYPE_SUITE"].value_counts()
total_occupations = len(Current_Application_data)
percentage_data = (occupation_data / total_occupations) * 100

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=NAME_TYPE_SUITE_data.index, y=NAME_TYPE_SUITE_data.values, palette="Set2")
for p, percentage in zip(ax.patches, percentage_data):
    ax.annotate(f"{percentage:.2f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center')
plt.xlabel("NAME_TYPE_SUITE")
plt.ylabel("Count")
plt.title("Distribution of NAME_TYPE_SUITE")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

missing_value_counts = Current_Application_data["NAME_TYPE_SUITE"].isnull().sum()
print("Missing Value Counts:")
print(missing_value_counts)
print("_______________________________________________________________________________")
unique_values = Current_Application_data["NAME_TYPE_SUITE"].unique()
print("Unique values for NAME_TYPE_SUITE:")
print(unique_values)
print("_______________________________________________________________________________")


### Obserbation: Most o the Applicants were Unaccompanied 

#### Checking the Missing Values and outliers in AMT_GOODS_PRICE 

Current_Application_data["AMT_GOODS_PRICE"].describe()

### Find the outliers by SCATTER PLOT

import seaborn as sns
import matplotlib.pyplot as plt

# Create a Seaborn relplot for a scatter plot
plt.figure(figsize=(10, 6))
sns.set_style("whitegrid")
scatter = sns.relplot(x=range(len(Current_Application_data)), y="AMT_GOODS_PRICE", data=Current_Application_data, kind="scatter", height=5, aspect=2)

plt.title("S AMT_GOODS_PRICE")
plt.xlabel("Index")
plt.ylabel("AMT_GOODS_PRICE")
plt.show()


missing_value_counts = Current_Application_data["AMT_GOODS_PRICE"].isnull().sum()
print("Missing Value Counts:")
print(missing_value_counts)
print("_______________________________________________________________________________")
unique_values_count = Current_Application_data["AMT_GOODS_PRICE"].nunique()
print("Unique count for NAME_TYPE_SUITE:")
print(unique_values_count)
print("_______________________________________________________________________________")

### Obserbation:  We have a lot of values were it preferable to convert to change to range of AMT_Price and do the analysis, for the missing value we can use IQR

#### Data study and observations in Current_Application_data is Completed and data is ready for further analysis

*_________________________________________________________________________________________________________________________*

### Univariate Analysis Current_Application_data

##  FLAG_cols

prefix = 'FLAG'
FLAG_cols = list(col for col in Current_Application_data.columns if col.startswith(prefix))
FLAG_cols

flag_data_relation = Current_Application_data[FLAG_cols + ["TARGET"]]
# to understand the plot we are currently taking Target value and Replacing 0 with "Non Defaulter" and 1 with "Defaulter" in the "TARGET" column
flag_data_relation.loc[flag_data_relation["TARGET"] == 0, "TARGET"] = "Non Defaulter"
flag_data_relation.loc[flag_data_relation["TARGET"] == 1, "TARGET"] = "Defaulter"
flag_data_relation.loc[flag_data_relation["FLAG_OWN_CAR"] == "N", "FLAG_OWN_CAR"] = 0
flag_data_relation.loc[flag_data_relation["FLAG_OWN_CAR"] == "Y", "FLAG_OWN_CAR"] = 1
flag_data_relation.loc[flag_data_relation["FLAG_OWN_REALTY"] == "N", "FLAG_OWN_REALTY"] = 0
flag_data_relation.loc[flag_data_relation["FLAG_OWN_REALTY"] == "Y", "FLAG_OWN_REALTY"] = 1
flag_data_relation.loc[flag_data_relation["FLAG_OWN_CAR"] == "N", "FLAG_OWN_CAR"] = 0
flag_data_relation.loc[flag_data_relation["FLAG_OWN_CAR"] == "Y", "FLAG_OWN_CAR"] = 1
flag_data_relation.head(5)


flag_data_relation.shape


plt.figure(figsize=[21,25])

for i, column in enumerate(FLAG_cols, start=1):
    plt.subplot(7,4, i)
    sns.countplot(data=flag_data_relation, x=column, hue="TARGET")
    plt.grid(True)

plt.show()

### Observation
____________________
With Regards to the visualization above, we observed that FLAG_DOCUMENT_3, FLAG_EMP_PHONE,FLAG_OWN_REALTYhave some relation ,and FLAG_MOBIL ,FLAG_CONT_MOBILE  shows a strong positive response mostly all the applicants with FLAG_MOBIL/FLAG_CONT_MOBILE has good frequency , one  with the loan target achievement. Therefore, can be a valuable column for analysis. FLAG_DOCUMENT_8 and FLAG_DOCUMENT_6 very mild positive responses.we can omit  the remaining columns

FLAG_selected_cols = ['FLAG_DOCUMENT_3','FLAG_EMP_PHONE','FLAG_OWN_REALTY',]

_Now we check the rest of the coulumns that has highrelatiopnship with the loan Target_ 

EXT_DF = Current_Application_data.loc[:, ['EXT_SOURCE_2', 'EXT_SOURCE_3', 'TARGET']]

(EXT_DF.isnull().mean().round(2))*100
#EXT_SOURCE_3  contains 19.825307 null values.
#Just to find the relation :first impute the Null values and then we need to find the correlation between Targets.

####  Without changing the orginal data we could just find the 


### Observation 
_We could see there are not much diffrence in the outliers hence we can use the mean to impute the null values_

EXT_DF['EXT_SOURCE_2'].fillna(EXT_DF['EXT_SOURCE_2'].mean(), inplace=True)
EXT_DF['EXT_SOURCE_3'].fillna(EXT_DF['EXT_SOURCE_3'].mean(), inplace=True)

(EXT_DF.isnull().mean().round(2))*100

#Correlation EXT_DF
correlation_matrix = EXT_DF.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title('Correlation Heatmap of EXT_DF')
plt.show()


## Observation :
###### We could confirm that there is not much of correlation with EXT_SOURCE_2,EXT_SOURCE_3 on TARGET._



EXT_cols = ["EXT_SOURCE_2","EXT_SOURCE_3"]

We currently have 7 columns with missing values, and our next step is to appropriately address and impute these missing values

Current_Application_data["OCCUPATION_TYPE"].fillna("Undefined", inplace=True)

*Now we need to work on the Number of enquiries to Credit Bureau data in AMT_cols.Null values is ideal to be replaced with median value for the AMT_cols data.*

AMT_cols =Current_Application_data[['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE']].columns
AMT_cols

Current_Application_data[AMT_cols].describe()

#### Converting the numerical AMT_cols to the categorical for analysis

# Create a column with the income range
bins = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, float('inf')]
labels = ["0-100K", "100K-200K", "200K-300K", "300K-400K", "400K-500K", "500K-600K", "600K-700K", "700K-800K", "800K-900K", "900K-1M", "1M+"]
Current_Application_data['AMT_INCOME_RANGE'] = pd.cut(Current_Application_data['AMT_INCOME_TOTAL'], bins, labels=labels)


Current_Application_data['AMT_INCOME_RANGE'].value_counts()

Current_Application_data['AMT_INCOME_RANGE'].value_counts(normalize=True).plot.bar()
plt.show()

# Create a column with the range Credit loan amount
bins = [0, 100000, 200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000, 1000000, float('inf')]
labels = ["0-100K", "100K-200K", "200K-300K", "300K-400K", "400K-500K", "500K-600K", "600K-700K", "700K-800K", "800K-900K", "900K-1M", "1M+"]
Current_Application_data['AMT_CREDIT_RANGE'] = pd.cut(Current_Application_data['AMT_CREDIT'], bins, labels=labels)


Current_Application_data.AMT_CREDIT_RANGE.value_counts()

Current_Application_data['AMT_CREDIT_RANGE'].value_counts(normalize=True).plot.bar()
plt.show()

# Create a  column with the range Loan annuity
bins = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, float('inf')]
labels = ["0-10K", "10K-20K", "20K-30K", "30K-40K", "40K-50K", "50K-60K", "60K-70K", "70K-80K", "80K-90K", "90K-100k", "100K+"]
Current_Application_data['AMT_ANNUITY_RANGE'] = pd.cut(Current_Application_data['AMT_ANNUITY'], bins, labels=labels)


Current_Application_data['AMT_ANNUITY_RANGE'].value_counts()

Current_Application_data['AMT_ANNUITY_RANGE'].value_counts(normalize=True).plot.bar()
plt.show()

# Create a column with the range goods price
bins = [0, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, float('inf')]
labels = ["0-10K", "10K-20K", "20K-30K", "30K-40K", "40K-50K", "50K-60K", "60K-70K", "70K-80K", "80K-90K", "90K-100k", "100K+"]
Current_Application_data['AMT_GOODS_PRICE_RANGE'] = pd.cut(Current_Application_data['AMT_GOODS_PRICE'], bins, labels=labels)

Current_Application_data['AMT_GOODS_PRICE_RANGE'].value_counts()

Current_Application_data['AMT_GOODS_PRICE_RANGE'].value_counts(normalize=True).plot.bar()
plt.show()

Current_Application_data.describe()

From the above we could see that these columns have negative value check if the negative values are required if not convert the negative values to positive (DAYS_BIRTH,DAYS_EMPLOYED,DAYS_REGISTRATION,DAYS_ID_PUBLISH,DAYS_LAST_PHONE_CHANGE)

#converting the negative values to positive
neg_cols = ["DAYS_BIRTH", "DAYS_EMPLOYED", "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]
for col in neg_cols:
    Current_Application_data[col] = Current_Application_data[col].apply(np.abs)

# Create a column with the age group
Current_Application_data['AGE'] = Current_Application_data['DAYS_BIRTH'] // 365
bins = [0, 20, 30, 40, 50, 60, 100, float('inf')]
labels = ["0-20", "20-30", "30-40", "40-50", "50-60", "60-100", "100+"]
# Create the 'AGE_RANGE'
Current_Application_data['AGE_RANGE'] = pd.cut(Current_Application_data['AGE'], bins=bins, labels=labels)


Current_Application_data['AGE_RANGE'].value_counts()

Current_Application_data['AGE_RANGE'].value_counts(normalize=True).plot.bar()
plt.show()

# Create a column with the DAYS EMPLOYED
Current_Application_data['YEARS_EMPLOYED']= Current_Application_data['DAYS_EMPLOYED']//365


Current_Application_data["DAYS_REGISTRATION"].describe()

Current_Application_data.nunique().sort_values()

Current_Application_data[[ "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]].describe()
#Keeping this data as days as there is not much of scatering in the data

plot_cols=[ "DAYS_REGISTRATION", "DAYS_ID_PUBLISH", "DAYS_LAST_PHONE_CHANGE"]
fig, axes = plt.subplots(1, 3, figsize=(14, 8))
axes = axes.flatten()
i=0
for i, col in enumerate(plot_cols):
    sns.boxenplot(data=Current_Application_data[col],color="y", ax=axes[i])
    axes[i].set_title(f"Column: {col}")

Keeping this data as days as there is not much of scatering in the data values

Current_Application_data.head(10).sort_index()

### Datatype to Categorical 

selected_cols = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE', 'FLAG_OWN_REALTY', 'LIVE_CITY_NOT_WORK_CITY','REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_WORK_REGION','LIVE_REGION_NOT_WORK_REGION', 'REGION_RATING_CLIENT', 'WEEKDAY_APPR_PROCESS_START','REGION_RATING_CLIENT_W_CITY']
for i in selected_cols:
    Current_Application_data[i] =pd.Categorical(Current_Application_data[i])

Current_Application_data.info()


null_imp_list=list(Current_Application_data.columns[(Current_Application_data.isnull().sum())*100 > 0] )
null_imp_list
for i in null_imp_list:
    print("null count for",i,"is",Current_Application_data[i].isnull().sum()," and percentage to the whole data is",(Current_Application_data[i].isnull().mean().round(3))*100)

null_imp_list

NAME_TYPE_SUITE,AMT_GOODS_PRICE_RANGE,AMT_ANNUITY_RANGE null values are changed to mode as they are categorical


## Observation:

#### In Current_Application_data the columns AMT_GOODS_PRICE_RANGE its advised to use mode to impute
#### In Current_Application_data the columns AMT_ANNUITY_RANGE its advised to use mode to impute
#### In Current_Application_data the columns NAME_TYPE_SUITE its advised to use mode to impute

Current_Application_data.describe()

Current_Application_data[['AMT_ANNUITY', 'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_GOODS_PRICE', 'AGE','CNT_CHILDREN','YEARS_EMPLOYED']].describe()

## Observation:

####  In the case of the 'YEARS_EMPLOYED' variable, there is an outlier with a maximum value of 1000, which appears to be an incorrect input.

#### The variables 'AMT_ANNUITY,' 'AMT_CREDIT,' 'AMT_GOODS_PRICE,' and 'CNT_CHILDREN' contain some outliers, suggesting that certain data points fall outside the typical range.

#### 'AMT_INCOME_TOTAL' has a significant number of outliers, indicating that some loan applicants have considerably higher incomes compared to others.

#### The variable 'AGE' does not exhibit any outliers.

Current_Application_data.head()

Current_Application_data.describe()

# Data Analysis

### Univariate Analysis

### Target

target_counts = Current_Application_data["TARGET"].value_counts()
labels = ['Non-Defaulter', 'Defaulter']
colors = ['#66ff66', '#ff6666'] 

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('white')
plt.pie(target_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)

center_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(center_circle)

ax.axis('equal')

plt.title("Loan Repayment Status Proportion", fontsize=14, pad=20)
plt.legend(loc='upper center', labels=labels, fontsize=12, bbox_to_anchor=(0.5, -0.0))
plt.show()


### Observation 
####  From the above is 91.9% of the applicants were able to repay only 8.1% are unable.

 ### Defaulter and Non-Defaulters

# Change the values in the "TARGET" column in the original DataFrame
Current_Application_data["TARGET"].replace({0: "Defaulter"}, inplace=True)
Current_Application_data["TARGET"].replace({1: "Non-Defaulter"}, inplace=True)


Defaulter_DF=Current_Application_data[Current_Application_data['TARGET']=="Defaulter"]
Defaulter_DF.head()

Non_Defaulter_DF=Current_Application_data[Current_Application_data['TARGET']=="Non-Defaulter"]
Non_Defaulter_DF.head()

### Univatiate Categorical cols

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
Cat_col = ['FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_EDUCATION_TYPE']


plt.figure(figsize=(14, 5 * len(Cat_col)))

for i, column in enumerate(Cat_col, 1):
    plt.subplot(len(Cat_col), 2, 2 * i - 1)
    ax = sns.countplot(x=column, data=Defaulter_DF, palette="Set2")
    plt.title('Defaulters', fontsize=12)
    ax.set(xlabel=column)
    temp = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)

    plt.subplot(len(Cat_col), 2, 2 * i)
    ax = sns.countplot(x=column, data=Non_Defaulter_DF, palette="Set2")
    plt.title('Non-Defaulters', fontsize=12)
    ax.set(xlabel=column)
    temp = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)
plt.tight_layout()
plt.show()


### Observation : 
#### Educataion type on both defaulters and non defaulters is highest for Secoandary/secondary special.
#### The applicants who has no car tends to be defaulters. 

### Univariate_continous_col

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
Univariate_conti_col = ['NAME_CONTRACT_TYPE', 'CODE_GENDER', 'NAME_TYPE_SUITE', 'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE', 'WEEKDAY_APPR_PROCESS_START', 'ORGANIZATION_TYPE', 'FLAG_OWN_REALTY', 'LIVE_CITY_NOT_WORK_CITY', 'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY', 'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REGION_RATING_CLIENT', 'WEEKDAY_APPR_PROCESS_START', 'REGION_RATING_CLIENT_W_CITY']
plt.figure(figsize=(14, 5 * len(Univariate_conti_col)))

for i, column in enumerate(Univariate_conti_col, 1):
    if column=="ORGANIZATION_TYPE":
        continue
    plt.subplot(len(Univariate_conti_col), 2, 2 * i - 1)
    ax = sns.countplot(x=column, data=Defaulter_DF, palette="Set2")
    plt.title('Defaulters', fontsize=12)
    ax.set(xlabel=column)
    temp = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)

    plt.subplot(len(Univariate_conti_col), 2, 2 * i)
    ax = sns.countplot(x=column, data=Non_Defaulter_DF, palette="Set2")
    plt.title('Non-Defaulters', fontsize=12)
    ax.set(xlabel=column)
    temp = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)

plt.tight_layout()
plt.show()


### Observation:
#### Male has more Defaulters comparing to the female.
#### Region Rating under category 2 has more defaulters 
#### Housing type House / Appartment tends to have more applicants and defaulters.

### Segments - Univatiate

import matplotlib.pyplot as plt
import seaborn as sns

def create_dist_plot(data, column, title):
    plt.figure(figsize=(7, 5))
    sns.histplot(data=data, x=column, kde=True, palette="Set2")
    plt.title(title, fontsize=12)
    plt.xlabel(column)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.tight_layout()
    plt.show()
sns.set(style="whitegrid")

seg_cols = ["AGE_RANGE", "EXT_SOURCE_2", "AMT_CREDIT_RANGE", "AMT_INCOME_RANGE"]
for column in seg_cols:
    create_dist_plot(Defaulter_DF, column, "Defaulters")
    create_dist_plot(Non_Defaulter_DF, column, "Non-Defaulters")


### Observation : 
#### Age range of 30-40 has the highes Defaulters.
#### AMT Credit has highest Deafaulter with credits taken as 200k - 300k 
#### Amount income range 100k -200k has the most defaulters

### Univariate_ORGANIZATION_TYPE

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

ORGANIZATION_TYPE_col = ['ORGANIZATION_TYPE']

plt.figure(figsize=(14, 10))

for i, column in enumerate(ORGANIZATION_TYPE_col, 1):
    plt.subplot(len(ORGANIZATION_TYPE_col), 1, i)
    
    ax = sns.countplot(x=column, data=Defaulter_DF, palette="Set2")
    plt.title('Defaulters', fontsize=12)
    ax.set(xlabel=column)
    temp = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 10))

for i, column in enumerate(ORGANIZATION_TYPE_col, 1):
    plt.subplot(len(ORGANIZATION_TYPE_col), 1, i)
    
    ax = sns.countplot(x=column, data=Non_Defaulter_DF, palette="Set2")
    plt.title('Non-Defaulters', fontsize=12)
    ax.set(xlabel=column)
    temp = ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right', fontsize=10)

plt.tight_layout()
plt.show()


### Observation :
### The Organisation Type 3,Self-emplyed,XNA has highest number of Defaulters

## Cheking Imbalance Current_Application_data

### Target

target_counts = Current_Application_data["TARGET"].value_counts()
labels = ['Non-Defaulter', 'Defaulter']
colors = ['#66ff66', '#ff6666'] 

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('white')
plt.pie(target_counts, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90, pctdistance=0.85)

center_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(center_circle)

ax.axis('equal')

plt.title("Loan Repayment Status Proportion", fontsize=14, pad=20)
plt.legend(loc='upper center', labels=labels, fontsize=12, bbox_to_anchor=(0.5, -0.0))
plt.show()



### Observation 
####  From the above is 91.9% of the applicants were able to repay only 8.1% are unable.

###  categorical Columns

import matplotlib.pyplot as plt

for column in Current_Application_data.columns:
    if column == 'ORGANIZATION_TYPE':
        continue 

    if Current_Application_data[column].dtype == "category":
        value_counts = Current_Application_data[column].value_counts(normalize=True, dropna=False)
        unique_count = len(value_counts)

        if unique_count > 4:
            plt.figure(figsize=(8, 6))
            value_counts.plot(kind='bar')
            plt.title(column)
            plt.show()
            print(value_counts)
        else:
            plt.figure(figsize=(8, 8))
            wedges, texts, autotexts = plt.pie(value_counts, labels=value_counts.index, startangle=90, autopct='%1.1f%%')
            for i, text in enumerate(autotexts):
                text.set_text(f'{value_counts.index[i]}: {value_counts.iloc[i]*100:.1f}%')
            plt.axis('equal')
            plt.legend(value_counts.index, loc="center left", bbox_to_anchor=(1, 0.5), title=f'Distribution of {column}')
            plt.title(column)
            plt.show()
            print(value_counts)


### ORGANIZATION_TYPE

gh1 = plt.figure(figsize=(12, 8))
sns.countplot(x='ORGANIZATION_TYPE', data=Current_Application_data)
plt.xticks(rotation=90) 
plt.show()

### Imbalaning the Numerical Columns

int_columns = Current_Application_data.select_dtypes(include=['int'])

for column in int_columns:
    unique_values = Current_Application_data[column].unique()
    if column in ['CNT_CHILDREN', 'YEARS_EMPLOYED']:
        plt.figure(figsize=(8, 4))
        plt.title(f'Bar Plot of {column}')
        sns.barplot(x=Current_Application_data[column].value_counts().index, y=Current_Application_data[column].value_counts())
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()
    elif len(unique_values) <= 10:
        plt.figure(figsize=(6, 6))
        plt.title(f'Distribution of {column}')
        value_counts = Current_Application_data[column].value_counts()
        wedges, texts, autotexts = plt.pie(value_counts, labels=value_counts.index,autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.legend(value_counts.index, loc="center left", bbox_to_anchor=(1, 0.5), title=f'Distribution of {column}')
        plt.show()
    else:
        plt.figure(figsize=(8, 4))
        plt.title(f'Distribution of {column}')
        sns.histplot(Current_Application_data[column], kde=True)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.show()


### OCCUPATION_TYPE

occupation_data = Current_Application_data["OCCUPATION_TYPE"].value_counts()
total_occupations = len(Current_Application_data)
percentage_data = (occupation_data / total_occupations) * 100

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=occupation_data.index, y=occupation_data.values, palette="Set2")
for p, percentage in zip(ax.patches, percentage_data):
    ax.annotate(f"{percentage:.2f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center')
plt.xlabel("Occupation Type")
plt.ylabel("Count")
plt.title("Distribution of Occupation Types")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


### ORGANIZATION_TYPE

ORGANIZATION_TYPE = Current_Application_data["ORGANIZATION_TYPE"].value_counts()
total_organizations = len(Current_Application_data)
percentage_data = (ORGANIZATION_TYPE / total_organizations) * 100

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=ORGANIZATION_TYPE.index, y=ORGANIZATION_TYPE.values, palette="Set2")
for p, percentage in zip(ax.patches, percentage_data):
    ax.annotate(f"{percentage:.2f}%", (p.get_x() + p.get_width() / 2, p.get_height()), ha='center')
plt.xlabel("ORGANIZATION_TYPE")
plt.ylabel("Count")
plt.title("Distribution of Organization Types")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


### AMT_cols

AMT_cols = ['AMT_INCOME_RANGE', 'AMT_CREDIT_RANGE', 'AMT_ANNUITY_RANGE']
num_columns = len(AMT_cols)
num_rows = 1 
width_ratios = [3, 3, 3] 

fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 5), gridspec_kw={'width_ratios': width_ratios})

for i, column in enumerate(AMT_cols):
    ax = axes[i] 
    count_data = Current_Application_data.groupby([column, 'TARGET']).size().unstack()
    sns.barplot(x=count_data.index, y='Defaulter', data=count_data, palette='Set2', ax=ax)
    ax.set_title(f'Distribution of {column} with Target')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
plt.tight_layout()
plt.show()


Reg_time_cols = ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY', 'WEEKDAY_APPR_PROCESS_START', 
                'REG_REGION_NOT_WORK_REGION', 'LIVE_REGION_NOT_WORK_REGION', 'REG_CITY_NOT_LIVE_CITY', 
                'REG_CITY_NOT_WORK_CITY', 'LIVE_CITY_NOT_WORK_CITY']

num_columns = 3
num_rows = (len(Reg_time_cols) - 1) // num_columns + 1
fig, axes = plt.subplots(num_rows, num_columns, figsize=(16, 20))

for i, column in enumerate(Reg_time_cols):
    row = i // num_columns
    col = i % num_columns
    ax = axes[row, col]
    count_data = Current_Application_data.groupby([column, 'TARGET']).size().unstack()
    sns.barplot(x=count_data.index, y='Defaulter', data=count_data, palette='Set2', ax=ax)
    ax.set_title(f'Distribution of {column} with Target')
    ax.set_xlabel(column)
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    ax.legend(["Non-Defaulter", "Defaulter"])

if len(Reg_time_cols) % num_columns != 0:
    for i in range(len(Reg_time_cols), num_columns * num_rows):
        fig.delaxes(axes.flatten()[i])

plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

### AGE

AGE_data = Current_Application_data["AGE_RANGE"].value_counts()
total_occupations = len(Current_Application_data)

plt.figure(figsize=(12, 6))
ax = sns.barplot(x=AGE_data.index, y=AGE_data.values, palette="Set2")

plt.xlabel("AGE")
plt.ylabel("Count")
plt.title("Distribution of AGE Types")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

### Observation : The age group ranging from 30- 40 and 40-50 years old represents the largest segment of loan applicants. 

### CONTRACT_TYPE

CONTRACT_TYPE_count = Current_Application_data.groupby(['NAME_CONTRACT_TYPE', 'TARGET']).size().unstack()
ax = CONTRACT_TYPE_count.plot(kind='bar', figsize=(10, 6))
plt.title('Distribution of Contract among Defaulters and Non-Defaulters')
plt.xlabel('Contract')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(["Non-Defaulter", "Defaulter"])
total_defaulter_count = CONTRACT_TYPE_count['Defaulter'].sum()
for p in ax.patches:
    percentage = f"{100 * p.get_height() / total_defaulter_count:.1f}%"
    ax.annotate(percentage, (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=10, color='black')
plt.show()

### Observation 
##### we could see that unpaid loans are 8.2 % of the 90.2 % of the Cash loans 
##### we could also see  unpaid loans are 9.8 % of the 0.6 % of the Revolving  loans  

### GENDER

gender_target_counts = Current_Application_data.groupby(['CODE_GENDER', 'TARGET']).size().unstack()

gender_target_counts.plot(kind='bar', figsize=(5, 3))
plt.title('Distribution of Gender among Defaulters and Non-Defaulters')
plt.xlabel('Gender')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.legend(["Non-Defaulter", "Defaulter"])
plt.show()
total_female_count = Current_Application_data[Current_Application_data['CODE_GENDER'] == 'F'].shape[0]
female_defaulter_count = Current_Application_data[(Current_Application_data['CODE_GENDER'] == 'F') & (Current_Application_data['TARGET'] == 'Defaulter')].shape[0]

print("Total Female Count:", total_female_count)
print("Female Defaulter Count:", female_defaulter_count)

total_male_count = Current_Application_data[Current_Application_data['CODE_GENDER'] == 'M'].shape[0]
male_defaulter_count = Current_Application_data[(Current_Application_data['CODE_GENDER'] == 'M') & (Current_Application_data['TARGET'] == 'Defaulter')].shape[0]

print("Total Male Count:", total_male_count)
print("Male Defaulter Count:", male_defaulter_count)

percentage_female_defaulter = (female_defaulter_count / total_female_count) * 100

percentage_male_defaulter = (male_defaulter_count / total_male_count) * 100

print("Percentage of Female replay is :", percentage_female_defaulter, "%")
print("Percentage of Male replay is:", percentage_male_defaulter, "%")


### Observation : The Female applicants is observed to be repaying more compared to the male.
#### Total Female Count: 202452 
#### Female Defaulter Count: 188282
##### Total Male Count: 105059
##### Male Defaulter Count: 94404
##### Percentage of Female repaying is : 93.00081006855946 %
##### Percentage of Male repaying is: 89.85807974566671 %

## ___________________________________________________________________________

## Bi- Variate / Multivariate Analysis on Current_Application_data

### Categorical and Categorical Variable

Defaulter_DF=Current_Application_data[Current_Application_data['TARGET']=="Defaulter"]
Defaulter_DF.head()

Non_Defaulter_DF=Current_Application_data[Current_Application_data['TARGET']=="Non-Defaulter"]
Non_Defaulter_DF.head()

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

plt.figure(figsize=(18, 10))

plt.subplot(2, 1, 1)  
sns.boxplot(x='AMT_INCOME_RANGE', y='AMT_CREDIT', data=Defaulter_DF, showfliers=False, palette="Set2")
plt.title("Income Range vs. Amt Credited for Defaulters", fontsize=14)
plt.xlabel("Income Range", fontsize=12)
plt.ylabel("Amt Credited", fontsize=12)

plt.subplot(2, 1, 2)  
sns.boxplot(x='AMT_INCOME_RANGE', y='AMT_CREDIT', data=Non_Defaulter_DF, showfliers=False, palette="Set2")
plt.title("Income Range vs. Amt Credited for Non-Defaulters", fontsize=14)
plt.xlabel("Income Range", fontsize=12)
plt.ylabel("Amt Credited", fontsize=12)

plt.tight_layout()
plt.show()


### Observation : 
#### As the income range increases the Amount credit is related as expected.

corr_cols = ['AMT_INCOME_TOTAL','AMT_CREDIT','AMT_ANNUITY','AMT_GOODS_PRICE','AGE','EXT_SOURCE_2','EXT_SOURCE_3','REGION_RATING_CLIENT']


df_corr_Defaulter_DF = Defaulter_DF[corr_cols]
df_corr_Defaulter_DF.describe()

df_corr_Defaulter_DF.corr() 

sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
correlation_matrix = df_corr_Defaulter_DF.corr()
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": 0.7})
plt.title("Correlation Heatmap for Defaulters", fontsize=16)
plt.show()


### Columns with strong correlations among defaulters include:AMT_ANNUITY and AMT_GOODS_PRICE  ,AMT_CREDIT and AMT_ANNUITY AMT_CREDIT and AMT_GOODS_PRICE 




df_corr_Non_Defaulter_DF = Non_Defaulter_DF[corr_cols]
df_corr_Non_Defaulter_DF.describe()

df_corr_Non_Defaulter_DF.corr() 

sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
correlation_matrix = df_corr_Non_Defaulter_DF.corr()
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": 0.7})
plt.title("Correlation Heatmap for Non-Defaulters", fontsize=16)
plt.show()

### Columns with strong correlations among non defaulters include AMT_CREDIT and AMT_ANNUITY ,AMT_CREDIT and AMT_GOODS_PRICE and AMT_ANNUITY and AMT_GOODS_PRICE
 

## Bivariate analysis on categorical variable

categories = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE',
             'NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','AGE_RANGE','AMT_INCOME_RANGE']

plt.figure(figsize=(25,40))
k=0
for category in categories:
    k = k+1
    ax = plt.subplot(4,3,k)
    sns.boxplot(x = category, y = 'AMT_CREDIT', data=Defaulter_DF)
    temp = ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right')

## Observation :
### If you earn more money, you usually get a bigger loan,Loans that you can use repeatedly often come with lower credit amounts,Younger people usually get smaller loans than those in their middle or older years,Whether you're a man or a woman, own a car or property, your credit amount remains similar



plt.figure(figsize=(25,40))
k=0
for category in categories:
    k = k+1
    ax = plt.subplot(4,3,k)
    sns.boxplot(x = category, y = 'AMT_CREDIT', data=Non_Defaulter_DF)
    temp = ax.set_xticklabels(ax.get_xticklabels(), rotation = 45, horizontalalignment='right')

#### Whether you're a man or a woman, own a car or a house, the loan amount you get is about the same
#### unemployed /Married  have more loan credits compared to other.


## Income group and gender

import matplotlib.pyplot as plt
import seaborn as sns

pivot_table = Current_Application_data.pivot_table(values='TARGET', index='AMT_INCOME_RANGE', columns='CODE_GENDER', aggfunc='count')

plt.figure(figsize=(8, 6))
sns.barplot(x=pivot_table.index, y=pivot_table.sum(axis=1), palette='Set2')
plt.xlabel('AMT_INCOME_RANGE')
plt.ylabel('Count of Defaulters')
plt.title('Count of Defaulters by Income Range')
plt.xticks(rotation=45)
plt.show()


#### It's noticeable that across all income categories, men are more likely to experience loan defaults than women.

## Credit amount group and Income group

import matplotlib.pyplot as plt
import seaborn as sns

pivot_table = Current_Application_data.pivot_table(values='TARGET', index='AMT_CREDIT_RANGE', columns='AMT_INCOME_RANGE', aggfunc='count')

plt.figure(figsize=(8, 6))
sns.barplot(x=pivot_table.index, y=pivot_table.sum(axis=1), palette='Set2')
plt.xlabel('Credit amount group')
plt.ylabel('Count of Defaulters')
plt.title('Count of Defaulters by Credit Amount Group')
plt.xticks(rotation=45)
plt.show()



#### 100k-200k income range are highly defaulted in all income groups.


## Age group and Income group


pivot_table = Current_Application_data.pivot_table(values='TARGET', index='AGE_RANGE', columns='AMT_INCOME_RANGE', aggfunc='count')
plt.figure(figsize=(12, 6))
sns.barplot(data=pivot_table, palette="Set2")
plt.xlabel('income')
plt.ylabel('Count of Defaulters')
plt.title('Count of Defaulters by Age Group and Income Range')
plt.xticks(rotation=45)
plt.legend(title='Income Range', loc='upper right', labels=pivot_table.columns)
plt.show()


## Profession and Gender

Current_Application_data.pivot_table(values='TARGET',index='NAME_INCOME_TYPE',columns='CODE_GENDER',aggfunc='count').plot.bar(figsize=(8,5),rot=90)
plt.xlabel('Profession')
plt.ylabel('Defaulters')

### Observation : Working people have more defaulters compared to the rest of the professions

## previous_data read and study the data

previous_data = pd. read_csv('previous_application.csv')

previous_data.shape

previous_data.info()



previous_data.describe()

### Handaling Null columns in previous data
_Get the null value percentage for each coloumns respectivily and decide whcich to drop_.

# Percentage of  missing values column wise
previous_data.isnull().mean().round(2)*100 

# Delete the columns having more than 40% missing values
drop_pcoloumns=previous_data.columns[previous_data.isnull().mean().round(2)*100>=40] 
drop_pcoloumns

len(drop_pcoloumns)


previous_data.drop(drop_pcoloumns,axis=1,inplace=True)


previous_data.shape



### From these 26 columns select the columns to use in our analysis. 


plt.figure(figsize=(8,4))  
plt.subplot(121)  
sns.histplot(previous_data['AMT_ANNUITY'], kde=True, color='green')
plt.title('AMT_ANNUITY Histogram')
plt.subplot(122)
sns.histplot(previous_data['AMT_GOODS_PRICE'], kde=True, color='green')
plt.title('AMT_GOODS_PRICE Histogram')
plt.tight_layout()
plt.show()

We  could see in the first fig there is only is high frequecy in the begining and might have outliers and its better to use median insted of mean.the second fig we could see frequecy change in along the x axis and its advisable to use mode.

##### Advised to impute missing values in 'AMT_ANNUITY' with the median
##### Advised to impute missing valuesin 'AMT_GOODS_PRICE' with the mode


##### CNT_PAYMENT is a major data value that we cannot impute withput the exact value hence we replace the null value with zero.
_CNT_PAYMENT :Term of previous credit at application of the previous application (CNT_PAYMENT null count is 38375)_

### Imbalancing in category Columns

previous_data.info()

CONTRACT = previous_data["NAME_CONTRACT_STATUS"].value_counts()
labels = ['Approved', 'Refused', 'Unused offer', 'Canceled']
colors = ['#66ff66', '#ff6666', '#6666ff', '#ff66ff']

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('white')
wedges, _, _ = plt.pie(CONTRACT, labels=labels, colors=colors, startangle=90, pctdistance=0.85, autopct='')

center_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(center_circle)

ax.axis('equal')

plt.title("Loan Contract Status Proportion", fontsize=14, pad=20)

legend_labels = [f'{label} ({percentage:.1f}%)' for label, percentage in zip(labels, CONTRACT / CONTRACT.sum() * 100)]
plt.legend(wedges, legend_labels, loc='upper center', fontsize=12, bbox_to_anchor=(0.5, -0.0))

plt.show()


null_prev = list((previous_data.isnull().sum()) > 0)
null_P_list=list(previous_data.columns[null_prev])
null_P_list
for i in null_P_list:
    print("null count for",i,"is",previous_data[i].isnull().sum(),"percentage to the whole data is",(previous_data[i].isnull().mean().round(3))*100)

###### convert DAYS_DECISION  in to year (Description:Relative to current application when was the decision about previous application made )

previous_data['DAYS_DECISION'] = previous_data['DAYS_DECISION'].abs()

previous_data['DAYS_DECISION_MONTHS'] = (previous_data['DAYS_DECISION'] / 30).astype(int)

previous_data.columns


col_list=['AMT_ANNUITY', 'AMT_CREDIT','AMT_APPLICATION', 'AMT_GOODS_PRICE','DAYS_DECISION', 'SELLERPLACE_AREA','CNT_PAYMENT','DAYS_DECISION_MONTHS']

fig, axes = plt.subplots(4,2,figsize=(10,18))

for col, ax in zip(col_list, axes.ravel()):
    sns.boxplot(y=previous_data[col], color='red', ax=ax)
    ax.set_title(col)
for i in range(8, 4 * 2):
    fig.delaxes(axes.ravel()[i])

#### Observation : CNT_PAYMENT and DAY_DECISION_ Month has data that we could rellay on analysis

col_list = ['AMT_ANNUITY', 'AMT_CREDIT', 'AMT_APPLICATION', 'AMT_GOODS_PRICE', 'DAYS_DECISION', 'SELLERPLACE_AREA', 'CNT_PAYMENT', 'DAYS_DECISION_MONTHS']

selected_data = previous_data[col_list]
selected_data.describe()



## Observation
###### CNT_PAYMENT and DAYS_DECISION have few outliers whereas rest of the coulumns 'AMT_ANNUITY', 'AMT_CREDIT','AMT_APPLICATION', 'AMT_GOODS_PRICE','SELLERPLACE_AREA' have large count in outiers.
###### SELLERPLACE_AREA has negative values might be an wrong data inputed.
###### AMT_ANNUITY,AMT_CREDIT,AMT_APPLICATION,AMT_GOODS_PRICE, we could use median to impute

## Converting the columns to Categorical

previous_cate_cols = ['NAME_CONTRACT_TYPE','NAME_CASH_LOAN_PURPOSE', 'NAME_CONTRACT_STATUS', 'NAME_PAYMENT_TYPE','CODE_REJECT_REASON', 'NAME_CLIENT_TYPE', 'NAME_GOODS_CATEGORY', 'NAME_PORTFOLIO','NAME_PRODUCT_TYPE', 'CHANNEL_TYPE', 'NAME_SELLER_INDUSTRY', 'NAME_YIELD_GROUP', 'PRODUCT_COMBINATION',]

for i in previous_cate_cols:
    previous_data[i]=pd.Categorical(previous_data[i])

 ### previous_cate_cols imbalance and outliners identified and catergorical colums converted .

# ________________________________________________________________

# Merge Data and Analysis

cols_current = ['SK_ID_CURR','TARGET','CODE_GENDER','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','AGE_RANGE','AMT_INCOME_RANGE','EXT_SOURCE_2','EXT_SOURCE_3']
# Creating a dataset from current application for merging 

Current_App_Merge_DF =Current_Application_data[cols_current]

Current_App_Merge_DF.shape

col_drop = ['SELLERPLACE_AREA','PRODUCT_COMBINATION','AMT_GOODS_PRICE']

previous_Merge_data = previous_data.drop(col_drop,axis=1)

previous_Merge_data.shape

MERGED_DF = pd.merge(previous_Merge_data,Current_App_Merge_DF, on='SK_ID_CURR', how='left')
MERGED_DF.head()

MERGED_DF.shape


nan_count = len(MERGED_DF[pd.isna(MERGED_DF['TARGET'])])
nan_count


 #### contains the 256513 of NaN values in the 'TARGET' column

MERGED_DF = MERGED_DF.dropna(subset=['TARGET'])


MERGED_DF['PERCENT_CREDIT'] = (MERGED_DF['AMT_CREDIT'] / MERGED_DF['AMT_APPLICATION'] * 100).round(2)
MERGED_DF.head()


MERGED_DF.info()

#"NAME_CASH_LOAN_PURPOSE",,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))

univariate_merged("NAME_CASH_LOAN_PURPOSE",L1,"NAME_CONTRACT_STATUS",["#548235","#FF0000","#0070C0","#FFFF00"],True,(18,7))

#  Univariate analysis Merged Data

### unordered vs categorical

### Contract Status Analysis

sns.set_palette("Set2")

CONTRACT = MERGED_DF["NAME_CONTRACT_STATUS"].value_counts()
labels = ['Approved', 'Refused', 'Unused offer', 'Canceled']

fig, ax = plt.subplots(figsize=(6, 6))
fig.patch.set_facecolor('white')
wedges, _, _ = plt.pie(CONTRACT, labels=labels, startangle=90, pctdistance=0.85, autopct='')

center_circle = plt.Circle((0, 0), 0.70, fc='white')
fig.gca().add_artist(center_circle)

ax.axis('equal')

plt.title("Contract Status", fontsize=14, pad=20)

legend_labels = [f'{label} ({percentage:.1f}%)' for label, percentage in zip(labels, CONTRACT / CONTRACT.sum() * 100)]
plt.legend(wedges, legend_labels, loc='upper center', fontsize=12, bbox_to_anchor=(0.5, -0.0))

plt.show()


### Observation of the total  applicants :
#### Approved % is 62% 
#### Refused % is 18%

### Client:

plt.figure(figsize=(8, 5))
ax = sns.countplot(x='NAME_CLIENT_TYPE', data=MERGED_DF)

ax.set_title('Clients', fontsize=16)
ax.set_xlabel('Client-Type', fontsize=12)
ax.set_ylabel('Count', fontsize=12)

ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
sns.set(style='whitegrid')
sns.set_palette("deep")


for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=12, color='black')

plt.tight_layout()
plt.show()


### Observation Client:The client type mostly were Repeaters who are applying for loan 

### Portfolio

plt.figure(figsize=(8, 5))
ax = sns.countplot(x='NAME_PORTFOLIO', data=MERGED_DF)

ax.set_title('Portfolios', fontsize=16)
ax.set_xlabel('Portfolio-Type', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')


sns.set(style='whitegrid')
sns.set_palette("deep")

for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=12, color='black')

plt.tight_layout()
plt.show()


### Observation Portfolio :
#### The majority of previous applications were for POS,  Cards and cars was relatively small.

### Channel 



plt.figure(figsize=(10, 6))
ax = sns.countplot(x='CHANNEL_TYPE', data=MERGED_DF)
ax.set_title('Application Channel Distribution', fontsize=16)
ax.set_xlabel('Application Channel', fontsize=12)
ax.set_ylabel('Count', fontsize=12)


ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')

sns.set(style='whitegrid')
sns.set_palette("Set3")


for p in ax.patches:
    ax.annotate(f'{p.get_height()}', (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=12, color='black')

plt.tight_layout()
plt.show()


### Observation:
#### Credit and Cash offices was the most commonly use type of channel, also Country-wide comes the second channel type.

### Applied loan amount

plt.figure(figsize=(10, 6))
sns.distplot(MERGED_DF['AMT_APPLICATION'], bins=20, hist=True, kde=True, color='skyblue')
plt.xlabel('Application Amount')
plt.ylabel('Density')
plt.title('Distribution Plot with Histogram and KDE of Application Amount')
plt.show()


### Obesevation:
#### the graph has a high peak in the start and the it decreases, mostly Applicants amount is below 250000 

### Credited loan amount

plt.figure(figsize=(10, 6))
sns.distplot(MERGED_DF['AMT_CREDIT'], bins=20, hist=True, kde=True, color='skyblue')
plt.xlabel('Credited Amount')
plt.ylabel('Frequency')
plt.title('Distribution Plot with Histogram of Credited Amount')
plt.show()


### Obesevation
#### Credited amount is also showing high peak in the start of the graph which show the range is below  250000 

### Decision Months

plt.figure(figsize=(10, 6))
sns.distplot(MERGED_DF['DAYS_DECISION_MONTHS'], bins=30, kde=True, color='blue')
plt.xlabel('Months')
plt.ylabel('Density')
plt.title('Distribution of Decision Months')
plt.show()

### Obesevation :mosly it takes 30 months 

## Bivariate analysis

 ###   Correlations Among Defaulters

sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
correlation_matrix = MERGED_DF.corr()
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": 0.7})
plt.title("Correlation Heatmap for Defaulters", fontsize=16)
plt.show()

 ###   Correlation Heatmap for Non-Defaulters

sns.set(style="whitegrid")
plt.figure(figsize=(10, 8))
correlation_matrix = MERGED_DF.corr()
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True, fmt=".2f", linewidths=0.5, square=True, cbar_kws={"shrink": 0.7})
plt.title("Correlation Heatmap for Non-Defaulters", fontsize=16)
plt.show()

## Observation:From the above two Corelation heatmap :
#### AMT_CREDIT shows high correlation with AMT_ANNUITY , AMT_APPLICATION 

## continious columns

plt.figure(figsize=(10, 6))
joint = sns.jointplot(data=MERGED_DF, x='AMT_APPLICATION', y='AMT_CREDIT', hue='NAME_CONTRACT_STATUS', kind='scatter')
joint.ax_joint.legend(loc='upper right', title='Contract Status')
plt.show()


## Observation 
#### Credited amount is related to the Application Amount .

 ## categorical variable:

MERGED_DF.info()

cols = ['NAME_CONTRACT_TYPE', 'NAME_CONTRACT_STATUS', 'NAME_CLIENT_TYPE', 'NAME_PORTFOLIO', 'CHANNEL_TYPE', 'AMT_INCOME_RANGE']
category = 'AMT_CREDIT'

num_cols = 2

num_rows = (len(cols) + num_cols - 1) // num_cols

plt.figure(figsize=(15, 10))

for i, col in enumerate(cols):
    plt.subplot(num_rows, num_cols, i + 1)
    ax = sns.barplot(x=col, y=category, data=MERGED_DF)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.title(f'Bar Plot for {category} by {col}')

plt.tight_layout()
plt.show()


## Observation:with amount credits
#### Contract type is more from Car loans
#### Client type is Mostly repeaters 
#### portfolio is sean common from Cars
#### Channel type commonly used is Car dealers 
#### Amount income range is mostly between 800k to 900k

## Loan Purpose 

Non_def_MERG_DF = MERGED_DF[MERGED_DF['TARGET']=="Non-Defaulter"]
Def_MERG = MERGED_DF[MERGED_DF['TARGET']=="Defaulter"]

MERGED_DF.head()

top_categories = Non_def_MERG_DF['NAME_CASH_LOAN_PURPOSE'].value_counts().nlargest(5).index

filtered_data = Non_def_MERG_DF[Non_def_MERG_DF['NAME_CASH_LOAN_PURPOSE'].isin(top_categories)]

plt.figure(figsize=(20, 10))
ax = sns.countplot(x='NAME_CASH_LOAN_PURPOSE', data=filtered_data, hue='NAME_CONTRACT_STATUS', order=top_categories, palette='viridis')

plt.ylabel("Count", fontsize=15)
plt.title('Top 5 Loan Purposes Non-Defaulters', fontsize=20)
plt.legend(loc="upper right")
plt.xticks(rotation=45, ha='right')
plt.grid(True)

total_counts = filtered_data['NAME_CASH_LOAN_PURPOSE'].count()

for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total_counts)
    x = p.get_x() + p.get_width() / 2.
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', fontsize=12, color='black', va='bottom')

plt.show()


top_categories = Def_MERG['NAME_CASH_LOAN_PURPOSE'].value_counts().nlargest(5).index

filtered_data = Def_MERG[Def_MERG['NAME_CASH_LOAN_PURPOSE'].isin(top_categories)]

plt.figure(figsize=(20, 10))
ax = sns.countplot(x='NAME_CASH_LOAN_PURPOSE', data=filtered_data, hue='NAME_CONTRACT_STATUS', order=top_categories, palette='viridis')

plt.ylabel("Count", fontsize=15)
plt.title('Top 5 Loan Purposes Defaulters', fontsize=20)
plt.legend(loc="upper right")
plt.xticks(rotation=45, ha='right')

plt.grid(True)
total_counts = filtered_data['NAME_CASH_LOAN_PURPOSE'].count()

for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total_counts)
    x = p.get_x() + p.get_width() / 2.
    y = p.get_height()
    ax.annotate(percentage, (x, y), ha='center', fontsize=12, color='black', va='bottom')
plt.show()


## Observation:
#### the most loan purpos is XPA and XNA
#### XNA has more Cancled than the approved which is to be higlighted for future references and Repair and other has more refused status.

### segmented :Status and Client type

pivot_table = MERGED_DF.pivot_table(index='NAME_CLIENT_TYPE', columns='NAME_CONTRACT_STATUS', aggfunc='size', fill_value=0)

pivot_table.plot(kind='bar', stacked=True, colormap='viridis')

plt.xlabel('Client Type', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.title('Client Type vs. Contract Status', fontsize=16)
plt.legend(title='Contract Status', loc='upper right')

plt.show()


### Observation:
#### :A Some of the repeat applicants have received rejection closly identify the reason , and there are very few instances of cancelled status among new applicants

#### AGE :AGE_RANGE                  

plt.figure(figsize=(8, 5))
ax = sns.countplot(x='NAME_CONTRACT_STATUS', hue='AGE_RANGE', data=MERGED_DF, palette='viridis')

plt.xlabel('Previous loan status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Age Range', loc='upper right')
plt.title('Loan Status by Age Range', fontsize=16)

plt.show()


### Observation : The age range of 30-40 exhibits the highest number of loan approvals, while individuals above the age of 60 receive fewer approvals.

#### Gender :CODE_GENDER

plt.figure(figsize=(8, 5))
ax = sns.countplot(x='NAME_CONTRACT_STATUS', hue='CODE_GENDER', data=MERGED_DF, palette='viridis')

plt.xlabel('Previous loan status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Age Range', loc='upper right')
plt.title('Loan Status by Gender', fontsize=16)

plt.show()


### Observation: Femals have more loan approaval than the male

#### Education :  NAME_EDUCATION_TYPE          

plt.figure(figsize=(8, 5))
ax = sns.countplot(x='NAME_CONTRACT_STATUS', hue='NAME_EDUCATION_TYPE', data=MERGED_DF, palette='viridis')

plt.xlabel('Previous loan status', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.legend(title='Age Range', loc='upper right')
plt.title('Loan Status Education', fontsize=16)

plt.show()


### Observation :Education Secondary and Secondary Special has more aapproval.

### Observation Insight

The age groups ranging from 30-40 and 40-50 years old represent the largest segments of loan applicants.

Females have a significantly higher repayment rate (93.01%) compared to males (89.85%), indicating a preference for female applicants in loan approvals.

Cash loans have an 8.2% unpaid rate among 90.2% of total applications, while revolving loans have a 9.8% unpaid rate among 0.6% of total applications. It's crucial to carefully assess applicants in these categories to reduce the risk of defaults.

A positive correlation between income and the amount credited suggests that considering an applicant's income level when approving loan eligibility is beneficial.

Repeat applicants make up the predominant client category, making it advisable to prioritize this group when approving loans. New and refreshed applicants are comparatively less likely to secure loans.

Past applications predominantly revolved around POS, with fewer applications for Cards and cars. Prioritizing POS and exploring loan providing platforms is recommended.

There is a correlation between the amount credited and the application amount, indicating that as the application amount increases, the credited amount also tends to increase.

The two most common loan purposes are XPA and XNA, with XNA having a higher number of canceled cases compared to approved cases, particularly in the categories Repair and Other.

Some repeat applicants have experienced rejection, making it essential to closely identify the reasons behind these rejections. Additionally, there are minimal cases of loan cancellations among new applicants.

Education levels classified as Secondary and Secondary Special show a higher approval rate, which can guide decisions related to applicant education levels.

### Observation Points to Keep in Mind:

Be cautious when dealing with clients who have previously experienced loan refusals, cancellations, or unused offers, as they may pose a higher risk.

Exercise prudence when considering low-income groups with a history of previous loan refusals, as their financial capacity may impact their ability to repay.

Clients without a source of employment should be assessed carefully, as their financial stability and repayment capacity may be uncertain.

Young clients, particularly in the age group of 20-40, tend to exhibit a higher risk profile compared to mid-age clients and senior citizens.

Applicants with lower secondary and secondary education levels may require additional scrutiny in the approval process, as they are associated with a higher risk of default.

