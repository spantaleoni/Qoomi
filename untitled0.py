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

# List of allowed execution types
allowed_execution_types = ['exec_listener', 'exec_listener_bridge', 'exec_sched', 'sub_process']


# Load the CSV data
data_path = "/home/simonlesflex/Boomi/Qoomi/EXERecords.csv"

df = pd.read_csv(data_path, sep=',', index_col="executionId", error_bad_lines=False)

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


# Example: Create a histogram of execution durations
plt.figure(figsize=(8, 6))
sns.histplot(df['executionDuration'], bins=20, kde=True)
plt.xlabel('Execution Duration')
plt.ylabel('Frequency')
plt.title('Distribution of Execution Durations')
plt.show()


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


from sklearn.ensemble import IsolationForest

# Replace 'N/A' with 0 (or any other appropriate value)
df['executionTime'] = df['executionTime'].replace('N/A', 0)
# Remove rows with missing values in the column
df = df.dropna(subset=['executionTime'])
# Fit an Isolation Forest model to identify outliers in execution duration
clf = IsolationForest(contamination=0.05)
df['is_outlier'] = clf.fit_predict(df[['executionDuration']])

import matplotlib.colors as mcolors

# Create a custom colormap for binary values (outlier or not)
# Convert the 'is_outlier' column to strings
df['is_outlier'] = df['is_outlier'].astype(int)

unique_values = df['is_outlier'].unique()
print(unique_values)


# Create a custom colormap for binary values (outlier or not)
cmap = mcolors.ListedColormap(['blue', 'red'])
df['is_outliertxt'] = df['is_outlier'].replace('-1', 'red')
df['is_outliertxt'] = df['is_outlier'].replace('1', 'blue')
# Visualize outliers using the custom colormap
plt.figure(figsize=(8, 6))
plt.scatter(df.index, df['executionDuration'], c=df['is_outliertxt'], cmap=cmap, marker='o', edgecolor='k')
# Add a colorbar to the plot
plt.colorbar(ticks=[0, 1], label='Outlier Status', format='%d')

# Add a colorbar to the plot
plt.colorbar(ticks=['0', '1'], label='Outlier Status')

plt.xlabel('Index')
plt.ylabel('Execution Duration')
plt.title('Outlier Detection in Execution Duration')
plt.show()



# Identify and analyze outliers
outlier_df = df[df['is_outlier'] == -1]



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
plt.show()


# Calculate the correlation matrix
correlation_matrix = df.corr()

# Visualize the correlation matrix as a heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
