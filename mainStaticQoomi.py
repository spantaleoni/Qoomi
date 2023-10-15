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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.stats import ttest_ind
from scipy.stats import zscore
import numpy as np
import sys

# List of allowed execution types
allowed_execution_types = ['exec_listener', 'exec_listener_bridge', 'exec_sched', 'sub_process']
BASE_PATH = '/home/simonlesflex/Boomi/Qoomi/'
BASE_PATHF = '/home/simonlesflex/Boomi/Qoomi/Fig/'
G_SUBSET = 100000000
G_TOPN_PROC = 30  # You can change this value to your desired number
G_TOPN_PROCDUR = 30
G_NODEthreshold = 2  # Define your threshold for anomaly detection
G_KMEANS_Clusters = 3
G_EXECDthreshold = 5000
# Set the execution duration and error rate thresholds
threshold_duration = 10000  # 10 seconds
threshold_error_rate = 2    # 2%

G_DISTRIBUTIONFlag = False
G_SCATTERFlag = False
G_SAVEFIG = True
G_CLUSTERRANGE_Flag = False
# Define a log file path
G_NODElog_file_path = 'NODEanomaly_log.txt'
G_DEPLOYEDUNEXEC_file_path = 'HNKDeployedUnexecuted_Processes2.csv'
G_DEPLOYEDUNIQUE_file_path = 'HNKDeployedUniqueID_Processes2.csv'
G_AVGDURATIONFile = BASE_PATHF + 'AvgDuration.jpg'
G_DISTRLOGDURATIONFile = BASE_PATHF + 'DistributionLogDuration.jpg'
G_HIGHEXECDURATIONFile = BASE_PATHF + 'HighExecutionDuration.jpg'
G_HIGHERRORRATEFile = BASE_PATHF + 'HighErrorRate.jpg'
G_DAYEXECCOUNTFile = BASE_PATHF + 'DailyExecutionCountOverTime.jpg'
G_TIMEBASEDALLFile = BASE_PATHF + 'TimeBasedDistributionAll.jpg'
G_TIMEBASEDOUTFile = BASE_PATHF + 'TimeBasedDistributionOutliers.jpg'
G_BOXDURATIONOUTFile = BASE_PATHF + 'BoxPlotDurationOutliers.jpg'
G_BOXDURATIONALLFile = BASE_PATHF + 'BoxPlotDurationAll.jpg'
G_KMEANSFile = BASE_PATHF + 'KMeansClustering.jpg'
G_HEATFILTFile = BASE_PATHF + 'CorrelationHeatmapPCAFiltered.jpg'
G_HEATALLFile = BASE_PATHF + 'CorrelationHeatmapALL.jpg'
G_SCATTERALLFile = BASE_PATHF + 'ScatterALL.jpg'

# Load the CSV data  HNK_ExecutionRecords_AtomID_829e5ce5-94b6-483a-b591-78ea97a33b91.csv
#data_path = "/home/simonlesflex/Boomi/Qoomi/EXERecords.csv"

# Check if two filenames were provided as arguments
if len(sys.argv) != 3 and len(sys.argv) != 1:
    print("Usage: python your_script.py filename1.csv filename2.csv")
    sys.exit(1)
else:
    if len(sys.argv) == 1:
        filename1 = 'HNK_ExecutionRecords_AtomID_829e5ce5-94b6-483a-b591-78ea97a33b91.csv'
        filename2 = 'HNK_DeployedProcesses_AtomID_829e5ce5-94b6-483a-b591-78ea97a33b91.csv'
    else:
        # Get the filenames from the command line arguments
        filename1 = sys.argv[1]
        filename2 = sys.argv[2]
    


data_path = BASE_PATH + filename1
deployed_processes_path = BASE_PATH + filename2


df = pd.read_csv(data_path, sep=',', index_col="executionId", error_bad_lines=False)

# Load the deployed process data
deployed_processesdf = pd.read_csv(deployed_processes_path)
unique_deployed_process_ids = deployed_processesdf['processId'].unique()
unique_executed_process_ids = df['processId'].unique()

deployedDF_Unique = deployed_processesdf[deployed_processesdf['processId'].isin(unique_deployed_process_ids)]
deployedDF_Unique.to_csv(G_DEPLOYEDUNIQUE_file_path)

undeployed_unexecuted_processes = set(unique_deployed_process_ids) - set(unique_executed_process_ids)
details_of_undeployed_unexecuted_processes = deployed_processesdf[deployed_processesdf['processId'].isin(undeployed_unexecuted_processes)]

# Check if there are undeployed and unexecuted processes
if not details_of_undeployed_unexecuted_processes.empty:
    # Specify the filename for the CSV file
    csv_filename = 'undeployed_unexecuted_processes.csv'

    # Save the DataFrame to a CSV file
    details_of_undeployed_unexecuted_processes.to_csv(G_DEPLOYEDUNEXEC_file_path, index=False)

    print(f"Undeployed and unexecuted processes have been saved to {csv_filename}")
else:
    print("No undeployed and unexecuted processes found.")

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
df['executionId'] = df.index
df['executionTime'] = pd.to_datetime(df['executionTime'], format='%Y%m%d %H%M%S.%f', errors='coerce')
df.index = df['executionTime']
df['executionTime'] = df.index
G_BeginDate = str(df.executionTime.min())[:10]
G_EndDate = str(df.executionTime.max())[:10]


# Resample data to a specific time frequency (e.g., daily or hourly)
daily_mean_duration = df['executionDuration'].resample('D').mean()
daily_error_rate = (df['inboundErrorDocumentCount'] / df['inboundDocumentCount']).resample('D').mean()

# Plot time series data
fig = plt.figure(figsize=(24, 12))
plt.plot(daily_mean_duration, label='Average Execution Duration')
plt.plot(daily_error_rate, label='Error Rate')
plt.xlabel('Date')
plt.ylabel('Metric Value')
plt.legend()
plt.title('Time Series Analysis of Execution Duration and Error Rate - ' + G_BeginDate + ' | ' + G_EndDate)
plt.show()
fig.savefig(G_AVGDURATIONFile)

# Example: Create a histogram of execution durations
'''plt.figure(figsize=(8, 6))
sns.histplot(df['executionDuration'], bins=20, kde=True)
plt.xlabel('Execution Duration')
plt.ylabel('Frequency')
plt.title('Distribution of Execution Durations')
plt.show()'''
# Example: Clean and visualize the distribution of execution durations
fig = plt.figure(figsize=(20, 12))
# 1. Data Cleaning: Remove outliers (e.g., values greater than a threshold)
cleaned_data = df[df['executionDuration'] <= G_EXECDthreshold]
# 2. Log Transformation: Apply a log transformation to handle skewness
log_data = np.log1p(cleaned_data['executionDuration'])
# 3. Density Estimation (KDE): Use KDE to visualize the distribution
sns.histplot(log_data, bins=20, kde=True, color='blue')
plt.xlabel('Log(Execution Duration)')
plt.ylabel('Density')
plt.title('Distribution of Log(Execution Durations) - ' + G_BeginDate + ' | ' + G_EndDate)
plt.show()
fig.savefig(G_DISTRLOGDURATIONFile)

#Dimensionality Reduction:
#If you have a large number of features, consider using dimensionality reduction techniques like Principal Component Analysis (PCA) 
#to reduce the number of dimensions while preserving the most important information. This can be helpful for visualizing high-dimensional data.
pca = PCA(n_components=2)
reduced_features = pca.fit_transform(df[['inboundDocumentCount', 'outboundDocumentCount', 'executionDuration']])





# Calculate average execution duration by process
average_duration_by_process = df.groupby('processName')['executionDuration'].mean()

# Calculate error rate by process
error_rate_by_process = (df.groupby('processName')['inboundErrorDocumentCount'].sum() / df.groupby('processName')['inboundDocumentCount'].sum()) * 100

# Identify processes with high execution durations or error rates
processes_to_improve_duration = average_duration_by_process[average_duration_by_process > threshold_duration]
processes_to_improve_error_rate = error_rate_by_process[error_rate_by_process > threshold_error_rate]

# Plotting average execution duration for processes with high durations
sorted_df = processes_to_improve_duration.sort_values(ascending=False)
processes_to_improve_duration = sorted_df[:G_TOPN_PROCDUR]

fig = plt.figure(figsize=(24, 12))
plt.bar(processes_to_improve_duration.index, processes_to_improve_duration.values)
plt.xticks(rotation=90)
plt.xlabel('Process Name')
plt.ylabel('Average Execution Duration (ms)')
plt.title('Processes with High Execution Durations - ' + G_BeginDate + ' | ' + G_EndDate)
plt.tight_layout()
plt.show()
fig.savefig(G_HIGHEXECDURATIONFile)

# Plotting error rates for processes with high error rates
fig = plt.figure(figsize=(24, 12))
plt.bar(processes_to_improve_error_rate.index, processes_to_improve_error_rate.values)
plt.xticks(rotation=90)
plt.xlabel('Process Name')
plt.ylabel('Error Rate (%)')
plt.title('Processes with High Error Rates - ' + G_BeginDate + ' | ' + G_EndDate)
plt.tight_layout()
plt.show()
fig.savefig(G_HIGHERRORRATEFile)


# Convert 'executionTime' to a datetime format
#df['executionTime'] = pd.to_datetime(df['executionTime'], format='%Y%m%d %H%M%S.%f', errors='coerce')

# Create a time series plot of execution counts over time
fig = plt.figure(figsize=(24, 12))
df.set_index('executionTime').resample('D').size().plot()
plt.xlabel('Date')
plt.ylabel('Execution Count')
plt.title('Daily Execution Count Over Time - ' + G_BeginDate + ' | ' + G_EndDate)
plt.show()
fig.savefig(G_DAYEXECCOUNTFile)


# Replace 'N/A' with 0 (or any other appropriate value)
#df['executionTime'] = df['executionTime'].replace('N/A', 0)

# Fit an Isolation Forest model to identify outliers in execution duration
#clf = IsolationForest(contamination=0.05)
clf = IsolationForest(contamination=0.10)
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
contains_numbers = df['nodeId'][-1][:3].isnumeric()
print(contains_numbers)  # True
if contains_numbers is True:
    df['nodeIdtxt'] = df['nodeId'].str[:13]
else:
    df['nodeIdtxt'] = df['nodeId'].str[:6]
df = df.dropna(subset=['nodeIdtxt'])
nodeIdtxt = df['nodeIdtxt'].unique()
print(nodeIdtxt)

# Visualize outliers using the custom colormap
if G_SCATTERFlag is True:
    fig = plt.figure(figsize=(24, 16))
    plt.scatter(df['nodeIdtxt'], df['executionDuration'], c=df['is_outliertxt'], cmap=cmap)
    # Add a colorbar to the plot
    plt.colorbar(label='Outlier Status', format='%d')
    plt.xlabel('nodeId')
    plt.ylabel('Execution Duration')
    plt.title('Outlier Detection in Execution Duration - ' + G_BeginDate + ' | ' + G_EndDate)
    plt.show()
    fig.savefig(G_SCATTERALLFile)



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
ALL DATA'''
# Group outliers by hour of the day
df['hour_of_day'] = df.index.hour
df_hourly_mean = df.groupby('hour_of_day')['executionDuration'].mean()

# Plot the hourly mean execution duration of outliers
fig = plt.figure(figsize=(20, 12))
plt.plot(df_hourly_mean)
plt.xlabel('Hour of Day')
plt.ylabel('Mean Execution Duration')
plt.title('Time-based Analysis of ALL Executions - ' + G_BeginDate + ' | ' + G_EndDate)
plt.xticks(range(24))
plt.grid(True)
plt.show()
fig.savefig(G_TIMEBASEDALLFile)
    
'''Time-based Analysis:
Since you have timestamps in 'executionTime', you can analyze when these outliers occur:'''
# Group outliers by hour of the day
outlier_df['hour_of_day'] = outlier_df.index.hour
outlier_hourly_mean = outlier_df.groupby('hour_of_day')['executionDuration'].mean()

# Plot the hourly mean execution duration of outliers
fig = plt.figure(figsize=(20, 12))
plt.plot(outlier_hourly_mean)
plt.xlabel('Hour of Day')
plt.ylabel('Mean Execution Duration of Outliers')
plt.title('Time-based Analysis of Outliers - ' + G_BeginDate + ' | ' + G_EndDate)
plt.xticks(range(24))
plt.grid(True)
plt.show()
fig.savefig(G_TIMEBASEDOUTFile)


'''Box Plots and Visualizations:
Create a box plot to compare the distribution of execution durations for outliers across different processes:'''
# Sort the DataFrame by 'executionDuration' in descending order and select the top 10 rows
top_outliers = outlier_df.sort_values(by='executionDuration', ascending=False).head(G_TOPN_PROCDUR*50)
top_all = df.sort_values(by='executionDuration', ascending=False).head(G_TOPN_PROCDUR*50)

# Box plot of execution duration by process name
fig = plt.figure(figsize=(24, 12))
sns.boxplot(x='processName', y='executionDuration', data=top_all)
plt.xlabel('Process Name')
plt.ylabel('Execution Duration')
plt.title('Box Plot of Execution Duration for ALL by Process - ' + G_BeginDate + ' | ' + G_EndDate)
plt.xticks(rotation=90)
plt.show()
fig.savefig(G_BOXDURATIONALLFile)



# Box plot of execution duration by process name
fig = plt.figure(figsize=(24, 12))
sns.boxplot(x='processName', y='executionDuration', data=top_outliers)
plt.xlabel('Process Name')
plt.ylabel('Execution Duration')
plt.title('Box Plot of Execution Duration for Outliers by Process - ' + G_BeginDate + ' | ' + G_EndDate)
plt.xticks(rotation=90)
plt.show()
fig.savefig(G_BOXDURATIONOUTFile)


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

if G_DISTRIBUTIONFlag is True:
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


'''K-MEANS CLUSTER: first need to preprocess the data and select the relevant columns for clustering. 
in this case, it's essential to choose numeric columns as k-means operates on numerical data. 
Here's the steps:

    Select the numeric columns from your DataFrame.
    Standardize the data (optional but recommended for k-means).
    Choose the number of clusters (k) based on your problem.
    Apply k-means clustering to the standardized data.
    Assign cluster labels to your DataFrame. '''

# Select numeric columns for clustering
knumeric_cols = ['inboundDocumentCount', 'outboundDocumentCount', 'executionDuration', 'inboundDocumentSize', 'outboundDocumentSize']

# Create a new DataFrame containing only the selected columns
knumeric_df = df[knumeric_cols]
# Standardize the data (recommended for k-means)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(knumeric_df)
# Choose the number of clusters (you need to specify 'k')
k = G_KMEANS_Clusters  # You can change this value based on your problem
# Apply k-means clustering
kmeans = KMeans(n_clusters=k, random_state=42)
cluster_labels = kmeans.fit_predict(scaled_data)
# Assign cluster labels to the original DataFrame
df['cluster'] = cluster_labels
# Choose two numeric columns for visualization (e.g., 'inboundDocumentCount' and 'executionDuration')
#x_feature = 'outboundDocumentCount'
x_feature = 'inboundDocumentSize'
y_feature = 'executionDuration'

# Create a scatter plot with color-coded clusters
fig = plt.figure(figsize=(30, 18))
for cluster_num in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster_num]
    plt.scatter(cluster_data[x_feature], cluster_data[y_feature], label=f'Cluster {cluster_num}')

plt.xlabel(x_feature)
plt.ylabel(y_feature)
plt.title('K-Means Clustering Results')
plt.legend()
plt.show()
fig.savefig(G_KMEANSFile)

# Perform k-means clustering with different numbers of clusters (k)
if G_CLUSTERRANGE_Flag is True:
    #cluster_range = range(2, 11)
    cluster_range = range((G_KMEANS_Clusters -1), G_KMEANS_Clusters)
    #cluster_range = G_KMEANS_Clusters
    cluster_scores = []
    
    for k in cluster_range:
    #k = cluster_range
        kmeans = KMeans(n_clusters=k, random_state=0)
        cluster_labels = kmeans.fit_predict(scaled_data)
        df['Cluster'] = cluster_labels
        silhouette_avg = silhouette_score(scaled_data, cluster_labels)
        cluster_scores.append(silhouette_avg)
        
    # Plot the silhouette scores for different values of k
    plt.figure(figsize=(10, 5))
    plt.plot(cluster_range, cluster_scores, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.title('Silhouette Score for Different Numbers of Clusters')
    plt.grid(True)
    plt.show()
    
    # Based on the silhouette scores, let's say we choose k=2 clusters as it has a high score

    # Perform k-means clustering with k=2

    # Analyze the clusters
    #cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=df.columns[:-1])
    cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=knumeric_cols)
    df['Cluster'].value_counts()

    # Decide which cluster to analyze further
    # Let's say we decide to analyze the cluster with the higher execution duration and document counts
    cluster_to_analyze = df.groupby('Cluster')[['executionDuration', 'inboundDocumentCount', 'outboundDocumentCount']].mean()
    #cluster_to_analyze['Cluster'] = cluster_to_analyze.idxmax(axis=0)
    cluster_to_analyze['Cluster'] = cluster_to_analyze.index

    # Filter the data for the chosen cluster
    cluster_data_to_analyze = df[df['Cluster'] == cluster_to_analyze['Cluster']]
    #cluster_data_to_analyze = df[df['Cluster'] == cluster_to_analyze]

    # Further analysis and actions can be applied to this cluster
    print("Cluster with Higher Execution Duration and Document Counts:")
    print(cluster_data_to_analyze)




# Create a pair plot
#sns.pairplot(filtered_df[numeric_cols])
#plt.show()

# Create a correlation matrix heatmap
'''correlation_matrix = df[numeric_cols].corr()
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix Heatmap')
plt.show() '''


# Calculate the correlation matrix
correlation_matrix = filtered_df[numeric_cols].corr()

# Create a heatmap of the correlation matrix
fig = plt.figure(figsize=(20, 12))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix FILTERED Heatmap - ' + G_BeginDate + ' | ' + G_EndDate)
plt.show()
fig.savefig(G_HEATFILTFile)


'''3. Detecting Execution Time Anomalies:
    Z-Score:

Calculate the z-score for execution times on each node and identify data points with z-scores exceeding a certain threshold.'''
# Group data by 'nodeIdtxt'
grouped_by_node = df.groupby('nodeIdtxt')


#for node, data in grouped_by_node:
#    z_scores = zscore(data['executionDuration'])
#    anomalies = data[abs(z_scores) > G_NODEthreshold]
#    # Process anomalies or log them for further investigation


'''IQR (Interquartile Range):

Use the IQR method to detect anomalies by defining a range within which most data points fall.'''
for node, data in grouped_by_node:
    q1 = data['executionDuration'].quantile(0.25)
    q3 = data['executionDuration'].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    anomalies = data[(data['executionDuration'] < lower_bound) | (data['executionDuration'] > upper_bound)]
    # Process anomalies or log them for further investigation

'''Machine Learning Models:

Machine learning models isolation forests, one-class SVMs, or autoencoders to detect anomalies in execution times. 
These models can capture complex patterns in the data.'''
for node, data in grouped_by_node:
    model = IsolationForest(contamination=0.05)  # Define your contamination parameter
    model.fit(data[['executionDuration']])
    anomalies = data[model.predict(data[['executionDuration']]) == -1]
    # Process anomalies or log them for further investigation


for node, data in grouped_by_node:
    # Detect anomalies using the chosen method (e.g., z-score)
    z_scores = zscore(data['executionDuration'])
    anomalies = data[abs(z_scores) > G_NODEthreshold]

    # Open the log file in append mode for each node's anomalies
    with open(G_NODElog_file_path, 'a') as log_file:
        # Add a heading for each node's anomalies
        log_file.write(f"=== Anomalies Detected on Node {node} ===\n")

        if not anomalies.empty:
            # Calculate the top 5 processName IDs by anomalies count
            top_process_names = anomalies['processName'].value_counts().head(5).index.tolist()

            # Include processName in the statistics header
            log_file.write(f"Total Anomalies: {len(anomalies)}\n")
            log_file.write(f"Top 5 Process Names by Anomalies Count: {', '.join(top_process_names)}\n")
            log_file.write(f"Average Execution Duration of Anomalies: {anomalies['executionDuration'].mean()} seconds\n")
            log_file.write(f"Maximum Execution Duration of Anomalies: {anomalies['executionDuration'].max()} seconds\n")
            log_file.write(f"Minimum Execution Duration of Anomalies: {anomalies['executionDuration'].min()} seconds\n")

            # Log individual anomalies with processName
            for index, row in anomalies.iterrows():
                log_file.write(f"Anomaly detected - ProcessName: {row['processName']} - ExecutionTime: {row['executionTime']} - ExecutionDuration: {row['executionDuration']} seconds\n")
        else:
            log_file.write("No anomalies detected on this node.\n")






df['executionId'] = df.index

corrdf = df.drop(['atomName', 'inboundDocumentCount', 'originalExecutionId', 'account', 'status', 'executionType', 
                  'processId', 'atomId', 'message', 'parentExecutionId', 
                  'topLevelExecutionId', 'launcherID', 'reportKey', 
                  'is_outlier', 'is_outliertxt', 'executionId', 'nodeId', 'nodeIdtxt', 'processName'], axis=1)

corrdf.index = corrdf['executionTime']
corrdf = corrdf.drop(['executionTime'], axis=1)
# Calculate the correlation matrix
correlation_matrix = corrdf.corr()

# Visualize the correlation matrix as a heatmap
fig = plt.figure(figsize=(20, 16))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap - ' + G_BeginDate + ' | ' + G_EndDate)
plt.show()
fig.savefig(G_HEATALLFile)
