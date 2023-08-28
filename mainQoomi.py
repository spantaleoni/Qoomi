#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 12:33:17 2023

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

# Convert "executionTime" to numeric format
df = df.dropna(subset=['executionTime'])
df['executionTime'] = pd.to_datetime(df['executionTime'], format='%Y%m%d %H%M%S.%f', errors='coerce')

df['Process Category'] = df['parentExecutionId'].apply(lambda x: 'Main Process' if pd.isna(x) else 'Child Process')

# Data Aggregation
agg_df = df.groupby(['executionType'])[['outboundDocumentCount', 'executionDuration',
                                                           'inboundDocumentSize', 'outboundDocumentSize']].mean()


# Boolean indexing to filter rows based on allowed execution types
agg_df = agg_df[agg_df.index.isin(allowed_execution_types)]


# K-Means Clustering
df = df.drop(columns={'originalExecutionId'})
kmeans_df = df[-500:][['outboundDocumentCount', 'executionDuration', 'inboundDocumentSize', 'outboundDocumentSize']].copy()
kmeans_df = (kmeans_df - kmeans_df.mean()) / kmeans_df.std()  # Standardize data
num_clusters = 4
kmeans = KMeans(n_clusters=num_clusters)
kmeans_df['cluster'] = kmeans.fit_predict(kmeans_df)


# Streamlit App
st.title("Boomi Runtime Data Analytics Dashboard")

# Display Aggregated Data
st.subheader("Aggregated Data")
st.write(agg_df)

# Display K-Means Clustering Results
st.subheader("K-Means Clustering Results")
st.write(kmeans_df)

# Visualization
st.subheader("Data Visualization")

# Sidebar for filtering options
st.sidebar.header('Filter Data')
process_category = st.sidebar.selectbox('Select Process Category', ['All', 'Main Process', 'Child Process'])
filtered_data = df if process_category == 'All' else df[df['Process Category'] == process_category]

# Interactive scatter plot
st.header('Execution Insights')
fig = px.scatter(filtered_data, x='executionTime', y='executionDuration', color='Process Category',
                 title='Execution Duration Over Time')
st.plotly_chart(fig)

# Interactive bar chart
st.header('Document Count Analysis')
#bar_data = filtered_data.groupby(['Process Category'])['outboundDocumentCount'].sum().reset_index()
doccnt_data = filtered_data.groupby(['outboundDocumentCount']).sum().reset_index()
doccnt_data['executionDuration'] = doccnt_data['executionDuration']/1000
#bar_fig = px.bar(bar_data, x='outboundDocumentCount', y='executionDuration', color='executionDuration',
#                 title='Outbound Document vs Execution Duration')
doccnt_fig = px.scatter(doccnt_data, x='outboundDocumentCount', y='executionDuration', color='executionDuration',
                     title='Outbound Document vs Execution Duration')
st.plotly_chart(doccnt_fig)
# Interactive bar chart
st.header('Document Size Analysis')
#bar_data = filtered_data.groupby(['Process Category'])['outboundDocumentCount'].sum().reset_index()
docsize_data = filtered_data.groupby(['outboundDocumentSize']).sum().reset_index()
docsize_data['executionDuration'] = docsize_data['executionDuration']/1000
#bar_fig = px.bar(bar_data, x='outboundDocumentCount', y='executionDuration', color='executionDuration',
#                 title='Outbound Document vs Execution Duration')
docsize_fig = px.scatter(docsize_data, x='outboundDocumentSize', y='executionDuration', color='executionDuration',
                     title='Outbound Document vs Execution Duration')
st.plotly_chart(docsize_fig)


# Bar Plot of Execution Type vs Outbound Document Count
fig_bar = plt.figure(figsize=(10, 6))
sns.barplot(data=agg_df.reset_index(), x='executionType', y='outboundDocumentCount', hue='outboundDocumentSize')
plt.title('Outbound Document Count by Execution Type')
st.pyplot(fig_bar)

# Scatter Plot of Clusters
fig_scatter = plt.figure(figsize=(10, 6))
sns.scatterplot(data=kmeans_df, x='executionDuration', y='outboundDocumentSize', hue='cluster', palette='Set1')
plt.title('K-Means Clustering: Execution Duration vs Outbound Document Size')
st.pyplot(fig_scatter)
