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

# Load the CSV data
data_path = "/home/simonlesflex/Downloads/ExeRecords.csv"
df = pd.read_csv(data_path)

# Convert "executionTime" to numeric format
df['executionTime'] = pd.to_datetime(df['executionTime']).astype(int)

# Create a Streamlit app
st.title("KMeans Clustering of Execution Time")

# Select the number of clusters
num_clusters = st.sidebar.slider("Select number of clusters:", min_value=2, max_value=10, value=3)

# Initialize KMeans
kmeans = KMeans(n_clusters=num_clusters, random_state=0)

# Fit the KMeans model
X = df[['executionTime']]
kmeans.fit(X)

# Add the cluster labels to the dataframe
df['cluster'] = kmeans.labels_

# Display the clusters and their centers
st.write("Cluster Centers:")
cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=['executionTime'])
cluster_centers['cluster'] = range(num_clusters)
cluster_centers['executionTime'] = pd.to_datetime(cluster_centers['executionTime'])
st.write(cluster_centers)

# Plot the clustered data
fig, ax = plt.subplots()
ax.scatter(df['executionTime'], df['cluster'], c=df['cluster'], cmap='rainbow')
ax.set_xlabel('Execution Time')
ax.set_ylabel('Cluster')
ax.set_title('KMeans Clustering of Execution Time')
st.pyplot(fig)