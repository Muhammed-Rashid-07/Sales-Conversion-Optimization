"""
This is a boilerplate pipeline 'training_pipeline'
generated using Kedro 0.19.1
"""
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def data_cleaning(raw):
    df=pd.DataFrame(raw)
    print(df.isna().sum()) #Checking missing values
    print(df.duplicated().sum()) #Checking duplicates
    print(df.info()) #Checking data types
    print(df.nunique()) #Checking unique values of numerical features

    #Checking various values of Categorical columns
    print("Unique_Values of age: ", df['age'].unique())
    print("Unique_Values of gender: ", df['gender'].unique()) 

    #Statistics of Data
    print(df.describe())

    #Univariate Analysis
    print(df[['Impressions', 'Clicks', 'Spent', 'Total_Conversion', 'Approved_Conversion']].describe())

    # Set up the plotting environment
    plt.figure(figsize=(16, 8))

    # Histograms for numerical variables
    plt.subplot(2, 3, 1)
    sns.histplot(df['Impressions'], bins=20, kde=True)
    plt.title('Distribution of Impressions')

    plt.subplot(2, 3, 2)
    sns.histplot(df['Clicks'], bins=20, kde=True)
    plt.title('Distribution of Clicks')

    plt.subplot(2, 3, 3)
    sns.histplot(df['Spent'], bins=20, kde=True)
    plt.title('Distribution of Spent')

    plt.subplot(2, 3, 4)
    sns.histplot(df['Total_Conversion'], bins=20, kde=True)
    plt.title('Distribution of Total_Conversion')

    plt.subplot(2, 3, 5)
    sns.histplot(df['Approved_Conversion'], bins=20, kde=True)
    plt.title('Distribution of Approved_Conversion')

    plt.tight_layout()
    plt.show()

    #Bivariate Analysis

    # Bar plots for categorical variables
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 3, 1)
    sns.countplot(x='age', data=df)
    plt.title('Distribution of Age')

    plt.subplot(1, 3, 2)
    sns.countplot(x='gender', data=df)
    plt.title('Distribution of Gender')

    plt.subplot(1, 3, 3)
    sns.countplot(x='interest', data=df)
    plt.title('Distribution of Interest')

    plt.tight_layout()
    plt.show()

    # Summary statistics for numerical variables
    print(df[['Impressions', 'Clicks', 'Spent', 'Total_Conversion', 'Approved_Conversion']].describe())

    # Set up the plotting environment
    plt.figure(figsize=(16, 8))

    # Histograms for numerical variables
    plt.subplot(2, 3, 1)
    sns.histplot(df['Impressions'], bins=20, kde=True)
    plt.title('Distribution of Impressions')

    plt.subplot(2, 3, 2)
    sns.histplot(df['Clicks'], bins=20, kde=True)
    plt.title('Distribution of Clicks')

    plt.subplot(2, 3, 3)
    sns.histplot(df['Spent'], bins=20, kde=True)
    plt.title('Distribution of Spent')

    plt.subplot(2, 3, 4)
    sns.histplot(df['Total_Conversion'], bins=20, kde=True)
    plt.title('Distribution of Total_Conversion')

    plt.subplot(2, 3, 5)
    sns.histplot(df['Approved_Conversion'], bins=20, kde=True)
    plt.title('Distribution of Approved_Conversion')

    plt.tight_layout()
    plt.show()



    # Bar plots for categorical variables
    plt.figure(figsize=(16, 8))

    plt.subplot(1, 3, 1)
    sns.countplot(x='age', data=df)
    plt.title('Distribution of Age')

    plt.subplot(1, 3, 2)
    sns.countplot(x='gender', data=df)
    plt.title('Distribution of Gender')

    plt.subplot(1, 3, 3)
    sns.countplot(x='interest', data=df)
    plt.title('Distribution of Interest')

    plt.tight_layout()
    plt.show()

    # Scatter plots for numerical variables

    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    sns.scatterplot(x='Impressions', y='Total_Conversion', data=df)
    plt.title('Total_Conversion vs Impressions')

    plt.subplot(2, 2, 2)
    sns.scatterplot(x='Clicks', y='Total_Conversion', data=df)
    plt.title('Total_Conversion vs Clicks')

    plt.subplot(2, 2, 3)
    sns.scatterplot(x='Spent', y='Total_Conversion', data=df)
    plt.title('Total_Conversion vs Spent')

    plt.subplot(2, 2, 4)
    sns.scatterplot(x='Impressions', y='Approved_Conversion', data=df)
    plt.title('Approved_Conversion vs Impressions')

    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    sns.boxplot(x='age', y='Total_Conversion', data=df)
    plt.title('Total_Conversion across Age Groups')

    plt.subplot(2, 2, 2)
    sns.boxplot(x='gender', y='Total_Conversion', data=df)
    plt.title('Total_Conversion across Genders')

    plt.subplot(2, 2, 3)
    sns.boxplot(x='age', y='Approved_Conversion', data=df)
    plt.title('Approved_Conversion across Age Groups')

    plt.subplot(2, 2, 4)
    sns.boxplot(x='gender', y='Approved_Conversion', data=df)
    plt.title('Approved_Conversion across Genders')

    plt.tight_layout()
    plt.show()
    # Demographic Analysis - Distribution of Age and Gender
    plt.figure(figsize=(16, 6))


    plt.tight_layout()
    plt.show()
   

    # Demographic Analysis
    # Demographic Analysis - Relationships with Ad Performance Metrics
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    sns.boxplot(x='age', y='Impressions', data=df)
    plt.title('Impressions across Age Groups')

    plt.subplot(2, 2, 2)
    sns.boxplot(x='age', y='Clicks', data=df)
    plt.title('Clicks across Age Groups')

    plt.subplot(2, 2, 3)
    sns.boxplot(x='gender', y='Impressions', data=df)
    plt.title('Impressions across Genders')

    plt.subplot(2, 2, 4)
    sns.boxplot(x='gender', y='Clicks', data=df)
    plt.title('Clicks across Genders')

    plt.tight_layout()
    plt.show()

    # Demographic Analysis - Relationships with Conversions
    plt.figure(figsize=(16, 8))

    plt.subplot(2, 2, 1)
    sns.boxplot(x='age', y='Total_Conversion', data=df)
    plt.title('Total_Conversion across Age Groups')

    plt.subplot(2, 2, 2)
    sns.boxplot(x='age', y='Approved_Conversion', data=df)
    plt.title('Approved_Conversion across Age Groups')

    plt.subplot(2, 2, 3)
    sns.boxplot(x='gender', y='Total_Conversion', data=df)
    plt.title('Total_Conversion across Genders')

    plt.subplot(2, 2, 4)
    sns.boxplot(x='gender', y='Approved_Conversion', data=df)
    plt.title('Approved_Conversion across Genders')

    plt.tight_layout()
    plt.show()

    #Feature Engineering
    data = df

    data.drop(['ad_id'],axis=1,inplace=True)
    data.drop(['fb_campaign_id'],axis=1,inplace=True)
    # Define the age groups
    age_groups = ['30-34', '35-39', '40-44', '45-49']

    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Use label encoding to replace values in 'Age_Group'
    data['Age_Group'] = label_encoder.fit_transform(data['age'])


    # Gender Encoding
    data['Gender_Code'] = data['gender'].map({'F': 0, 'M': 1})

    # Interaction Features
    data['Interaction_Imp_Clicks'] = data['Impressions'] * data['Clicks']

    # Spent per Click
    data['Spent_per_Click'] = data['Spent'] / data['Clicks']

    # Total Conversion Rate
    data['Total_Conversion_Rate'] = data['Total_Conversion'] / data['Clicks']

    # Age-Gender Interaction
    data['Age_Gender_Interaction'] = data['Age_Group'].astype(str) + '_' + data['gender']

    # Budget Allocation
    data['Budget_Allocation_Imp'] = data['Spent'] / data['Impressions']


    # Ad Performance Metrics
    data['CTR'] = data['Clicks'] / data['Impressions']
    data['Conversion_per_Impression'] = data['Total_Conversion'] / data['Impressions']

    print(data.head())
    
    #Correlation Analysis
    numerical_features = data.select_dtypes(include=['int64', 'float64'])

    # Set the size of the heatmap
    plt.figure(figsize=(15, 12))

    # Create a heatmap using the correlation matrix
    sns.heatmap(numerical_features.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

    # Show the heatmap
    plt.show()

    # Set a threshold for correlation
    threshold = 0.95
    correlation_matrix = numerical_features.corr()
 
    # Find and drop highly correlated features
    highly_correlated = (correlation_matrix.abs() >= threshold).sum()
    highly_correlated = highly_correlated[highly_correlated > 1].index

    df_filtered = numerical_features.drop(columns=highly_correlated)
    print(highly_correlated)

    
    # Set the size of the heatmap
    plt.figure(figsize=(15, 12))

    # Create a heatmap using the correlation matrix
    sns.heatmap(df_filtered.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

    # Show the heatmap
    plt.show()

    print('Correlation matrix is {}',correlation_matrix)
    print(numerical_features)
    

    #data.drop(['ad_id'],axis=1,inplace=True)

    # Define the age groups
    age_groups = ['30-34', '35-39', '40-44', '45-49']

    # Initialize the label encoder
    label_encoder = LabelEncoder()

    # Use label encoding to replace values in 'Age_Group'
    data['Age_Group'] = label_encoder.fit_transform(data['age'])


    # Gender Encoding
    data['Gender_Code'] = data['gender'].map({'F': 0, 'M': 1})

    # # Interaction Features
    data['Interaction_Imp_Clicks'] = data['Impressions'] * data['Clicks']

    # Spent per Click
    data['Spent_per_Click'] = data['Spent'] / data['Clicks']

    # Total Conversion Rate
    data['Total_Conversion_Rate'] = data['Total_Conversion'] / data['Clicks']


    # Budget Allocation
    data['Budget_Allocation_Imp'] = data['Spent'] / data['Impressions']


    # Ad Performance Metrics
    data['CTR'] = data['Clicks'] / data['Impressions']
    data['Conversion_per_Impression'] = data['Total_Conversion'] / data['Impressions']

    data.drop(['age'],axis=1,inplace=True)
    data.drop(['gender'],axis=1,inplace=True)
    data.fillna(0, inplace=True)
    # Replace infinite values with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    data.fillna(0, inplace=True)
    print(data.head())


    data.to_csv("data/02_processed/processed_data.csv")