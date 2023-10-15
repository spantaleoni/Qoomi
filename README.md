mainStaticQoomi.py


**Table of Contents** 

    Introduction
    Overview
    Requirements
    Usage
    Features
    Anomaly Detection
    License

**Introduction**

mainStaticQoomi.py is a Python script designed for the analysis and visualization of execution records and process data in Qoomi, an integration platform. This script provides insights into executed processes, identifies outliers, and performs various data analysis tasks.


**Overview**

    Author: Simon Lesflex
    Last Updated: September 9, 2023
    License: MIT

The script uses various Python libraries, such as Pandas, Scikit-learn, Seaborn, and Matplotlib, to load, process, and analyze data.


**Requirements**

Before using this script, make sure you have the following:

    Python 3
    Required Python libraries (you can install them using pip):
        Streamlit
        Pandas
        Scikit-learn
        Seaborn
        Matplotlib
        Plotly
        NumPy

**Usage**

You can run the script from the command line by providing two CSV file paths as arguments. These CSV files contain data related to executed processes. Here's how to run the script:

bash
python mainStaticQoomi.py filename1.csv filename2.csv

If you don't provide arguments, the script will use default filenames for data files.


**Features**

    Data Analysis: The script loads and analyzes execution records and process data from Qoomi.
    Time Series Analysis: It provides insights into execution duration and error rates over time.
    Data Visualization: The script generates visualizations, such as time series plots, histograms, box plots, and scatter plots.
    Outlier Detection: It identifies outliers in execution duration and provides methods for handling them.
    K-Means Clustering: The script performs K-Means clustering on execution data and provides cluster visualization.
    Correlation Analysis: It analyzes the correlation between different numeric features.
    Anomaly Detection: The script uses Z-Scores, IQR, and machine learning models to detect execution time anomalies.
    Customization: You can adjust various parameters and thresholds for your specific needs.


**Anomaly Detection**  

The script provides multiple methods for detecting execution time anomalies:

    Z-Score: Calculates z-scores and identifies data points with scores exceeding a specified threshold.
    IQR (Interquartile Range): Uses the IQR method to define a range for identifying anomalies.
    Machine Learning Models: Utilizes isolation forests, one-class SVMs, or autoencoders for anomaly detection.

The detected anomalies are logged for further investigation.


**License**

This script is licensed under the MIT License. You are free to use and modify it for your specific needs. If you find it useful, please consider contributing to the open-source community and sharing your improvements.
Happy data analysis and anomaly detection with Qoomi!
For questions or contributions, feel free to reach out to the author, Simon Lesflex.
