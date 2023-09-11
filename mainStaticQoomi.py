#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep  9 16:52:09 2023

@author: simonlesflex
"""


import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from dateutil import parser
import seaborn as sns
import plotly.express as px

import matplotlib.colors as mcolors
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from scipy.stats import ttest_ind


# List of allowed execution types
allowed_execution_types = ['exec_listener', 'exec_listener_bridge', 'exec_sched', 'sub_process']
G_SUBSET = 1000000
G_TOPN_PROC = 15  # You can change this value to your desired number

# Load the CSV data
data_path = "/home/simonlesflex/Boomi/Qoomi/EXERecords.csv"

df = pd.read_csv(data_path, sep=',', index_col="executionId", error_bad_lines=False)

#df = df[-G_SUBSET:]

# Display the first few rows of the DataFrame
print(df.head())

# Get basic information about the dataset
print(df.info())

# Summary statistics of numeric columns
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Handle missing values if needed
df = df.fillna(0)

# Remove rows with missing values in the column
df = df.dropna(subset=['executionTime'])
# Remove rows with missing values in the column
df = df.dropna(subset=['nodeId'])

# Set 'executionTime' as the DataFrame index with DatetimeIndex
df['executionTime'] = pd.to_datetime(df['executionTime'], format='%Y%m%d %H%M%S.%f', errors='coerce')
df.index = df['executionTime']



# Resample data to a specific time frequency (e.g., daily or hourly)
daily_mean_duration = df['executionDuration'].resample('D').mean()
daily_error_rate = (df['inboundErrorDocumentCount'] / df['inboundDocumentCount']).resample('D').mean()

# Plot time series data
plt.figure(figsize=(12, 6))
plt.plot(daily_mean_duration, label='Average Execution Duration')
plt.plot(daily_error_rate, label='Error Rate')
plt.xlabel('Date')
plt.ylabel('Metric Value')
plt.legend()
plt.title('Time Series Analysis of Execution Duration and Error Rate')
plt.show()

# Example: Create a histogram of execution durations
plt.figure(figsize=(8, 6))
sns.histplot(df['executionDuration'], bins=20, kde=True)
plt.xlabel('Execution Duration')
plt.ylabel('Frequency')
plt.title('Distribution of Execution Durations')
plt.show()

#Dimensionality Reduction:
#If you have a large number of features, consider using dimensionality reduction techniques like Principal Component Analysis (PCA) 
#to reduce the number of dimensions while preserving the most important information. This can be helpful for visualizing high-dimensional data.
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(df[['inboundDocumentCount', 'outboundDocumentCount', 'executionDuration']])



# Set the execution duration and error rate thresholds
threshold_duration = 10000  # 10 seconds
threshold_error_rate = 5    # 5%

# Calculate average execution duration by process
average_duration_by_process = df.groupby('processName')['executionDuration'].mean()

# Calculate error rate by process
error_rate_by_process = (df.groupby('processName')['inboundErrorDocumentCount'].sum() / df.groupby('processName')['inboundDocumentCount'].sum()) * 100

# Identify processes with high execution durations or error rates
processes_to_improve_duration = average_duration_by_process[average_duration_by_process > threshold_duration]
processes_to_improve_error_rate = error_rate_by_process[error_rate_by_process > threshold_error_rate]

# Plotting average execution duration for processes with high durations
plt.figure(figsize=(12, 6))
plt.bar(processes_to_improve_duration.index, processes_to_improve_duration.values)
plt.xticks(rotation=90)
plt.xlabel('Process Name')
plt.ylabel('Average Execution Duration (ms)')
plt.title('Processes with High Execution Durations')
plt.tight_layout()
plt.show()

# Plotting error rates for processes with high error rates
plt.figure(figsize=(12, 6))
plt.bar(processes_to_improve_error_rate.index, processes_to_improve_error_rate.values)
plt.xticks(rotation=90)
plt.xlabel('Process Name')
plt.ylabel('Error Rate (%)')
plt.title('Processes with High Error Rates')
plt.tight_layout()
plt.show()


# Convert 'executionTime' to a datetime format
df['executionTime'] = pd.to_datetime(df['executionTime'], format='%Y%m%d %H%M%S.%f', errors='coerce')

# Create a time series plot of execution counts over time
plt.figure(figsize=(12, 6))
df.set_index('executionTime').resample('D').size().plot()
plt.xlabel('Date')
plt.ylabel('Execution Count')
plt.title('Daily Execution Count Over Time')
plt.show()



# Replace 'N/A' with 0 (or any other appropriate value)
#df['executionTime'] = df['executionTime'].replace('N/A', 0)

# Fit an Isolation Forest model to identify outliers in execution duration
clf = IsolationForest(contamination=0.05)
df['is_outlier'] = clf.fit_predict(df[['executionDuration']])

# Create a custom colormap for binary values (outlier or not)
# Convert the 'is_outlier' column to strings
df['is_outlier'] = df['is_outlier'].astype(str)

unique_values = df['is_outlier'].unique()
print(unique_values)

# Define a dictionary to specify replacements
replace_dict = {'-1': 'red', '1': 'blue'}
# Create a custom colormap for binary values (outlier or not)
cmap = mcolors.ListedColormap(['blue', 'red'])
df['is_outliertxt'] = df['is_outlier'].replace(replace_dict)
unique_valuestxt = df['is_outliertxt'].unique()
print(unique_valuestxt)
df['nodeIdtxt'] = df['nodeId'].str[:6]
df = df.dropna(subset=['nodeIdtxt'])
nodeIdtxt = df['nodeIdtxt'].unique()
print(nodeIdtxt)

# Visualize outliers using the custom colormap
plt.figure(figsize=(30, 20))
plt.scatter(df['nodeIdtxt'], df['executionDuration'], c=df['is_outliertxt'], cmap=cmap)
# Add a colorbar to the plot
plt.colorbar(label='Outlier Status', format='%d')
plt.xlabel('nodeId')
plt.ylabel('Execution Duration')
plt.title('Outlier Detection in Execution Duration')
plt.show()



# Identify and analyze outliers
outlier_df = df[df['is_outlier'] == '-1']


'''Grouping Outliers by a Categorical Variable:

In your outlier_df, you can group outliers by the 'processName' variable to 
understand how different processes are associated with outlier behavior:'''
# Group outliers by 'processName'
outlier_grouped = outlier_df.groupby('processName').agg({
    'executionDuration': 'mean',
    'inboundErrorDocumentCount': 'mean',
    'outboundDocumentCount': 'mean'
}).reset_index()

# Sort processes by the mean execution duration of outliers
outlier_grouped = outlier_grouped.sort_values(by='executionDuration', ascending=False)
print(outlier_grouped)


'''Time-based Analysis:

Since you have timestamps in 'executionTime', you can analyze when these outliers occur:'''
# Group outliers by hour of the day
outlier_df['hour_of_day'] = outlier_df.index.hour
outlier_hourly_mean = outlier_df.groupby('hour_of_day')['executionDuration'].mean()

# Plot the hourly mean execution duration of outliers
plt.figure(figsize=(10, 6))
plt.plot(outlier_hourly_mean)
plt.xlabel('Hour of Day')
plt.ylabel('Mean Execution Duration of Outliers')
plt.title('Time-based Analysis of Outliers')
plt.xticks(range(24))
plt.grid(True)
plt.show()


'''Box Plots and Visualizations:
Create a box plot to compare the distribution of execution durations for outliers across different processes:'''
# Box plot of execution duration by process name
plt.figure(figsize=(12, 6))
sns.boxplot(x='processName', y='executionDuration', data=outlier_df)
plt.xlabel('Process Name')
plt.ylabel('Execution Duration')
plt.title('Box Plot of Execution Duration for Outliers by Process')
plt.xticks(rotation=90)
plt.show()


#Perform the t-test:
#Use the ttest_ind function from scipy.stats to perform the t-test. This function returns the t-statistic and p-value.
# Compare mean execution duration of outliers to the entire dataset
entire_dataset_duration = df['executionDuration']
outlier_duration = outlier_df['executionDuration']

t_stat, p_value = ttest_ind(outlier_duration, entire_dataset_duration)


'''Interpret the Results:
    If the p-value is less than your chosen significance level (e.g., 0.05), 
    you can reject the null hypothesis and conclude that there is a statistically 
    significant difference in mean execution duration between outliers and the entire dataset.
    If the p-value is greater than the significance level, you fail to reject the null hypothesis, 
    indicating that there is no statistically significant difference.'''
significance_level = 0.05

print('*******************************')
if p_value < significance_level:
    print(f"Reject the null hypothesis. p-value = {p_value:.4f}")
else:
    print(f"Fail to reject the null hypothesis. p-value = {p_value:.4f}")


'''Effect Size:
#In addition to the t-test, it's often useful to calculate the effect size, which measures the magnitude of the difference between groups. 
#Common effect size measures include Cohen's d or eta-squared for ANOVA. 
#This can provide additional insights into the practical significance of the difference.'''
# Calculate effect size (Cohen's d)
mean_difference = outlier_duration.mean() - entire_dataset_duration.mean()
pooled_std_dev = ((outlier_duration.std() ** 2 + entire_dataset_duration.std() ** 2) / 2) ** 0.5
effect_size = mean_difference / pooled_std_dev
print(f"Effect size (Cohen's d): {effect_size:.4f}")




'''Histograms for Categorical Data:

If you have categorical columns in your dataset, you can create histograms 
or bar charts to visualize the distribution of categories. 
This can help you understand the frequency of each category and identify any imbalances.'''
# Determine the top X processes with the highest number of executions
top_processes = df['processName'].value_counts().nlargest(G_TOPN_PROC).index.tolist()
#Filter the DataFrame:
filtered_df = df[df['processName'].isin(top_processes)]
#Create Histograms for Categorical Data:
categorical_cols = ['status', 'executionType', 'processName']

for col in categorical_cols:
    plt.figure(figsize=(10, 6))
    ax = filtered_df[col].value_counts().plot(kind='bar')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    plt.xlabel(col)
    plt.ylabel('Count')
    plt.title(f'Distribution of {col}')
    plt.show()


'''Pairwise Scatter Plots:

If you have multiple numeric variables, you can create pairwise scatter plots or 
correlation matrices to visualize relationships between variables.'''
#numeric_cols = ['inboundDocumentCount', 'inboundErrorDocumentCount', 'outboundDocumentCount', 'executionDuration']
numeric_cols = ['inboundErrorDocumentCount', 'outboundDocumentCount', 'executionDuration']

# Create a pair plot
#sns.pairplot(filtered_df[numeric_cols])
#plt.show()

# Create a correlation matrix heatmap
correlation_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()


# Calculate the correlation matrix
correlation_matrix = filtered_df[numeric_cols].corr()

# Create a heatmap of the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show()





















df['executionId'] = df.index

'''
import networkx as nx

# Create a directed graph of process dependencies
G = nx.DiGraph()

# Add edges based on parent and child relationships
for _, row in df.iterrows():
    if not pd.isnull(row['parentExecutionId']):
        G.add_edge(row['parentExecutionId'], row['executionId'])

# Visualize the process dependency graph
plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos, with_labels=False, node_size=20)
plt.title('Process Dependency Graph')
plt.show()'''

corrdf = df.drop(['atomName', 'inboundDocumentCount', 'originalExecutionId', 'account', 'status', 'executionType', 
                  'processId', 'atomId', 'message', 'parentExecutionId', 
                  'topLevelExecutionId', 'launcherID', 'reportKey', 
                  'is_outlier', 'is_outliertxt', 'executionId', 'nodeId', 'nodeIdtxt', 'processName'], axis=1)

corrdf.index = corrdf['executionTime']
corrdf = corrdf.drop(['executionTime'], axis=1)
# Calculate the correlation matrix
correlation_matrix = corrdf.corr()

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
