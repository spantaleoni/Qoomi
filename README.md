Sure, here's a README file that explains the code you provided:

# Boomi Runtime Data Analytics Dashboard

This is a Streamlit application designed to provide operational insights about a Boomi runtime based on Java, executing data integration processes between applications.

## Overview

The code creates a Streamlit app that analyzes and visualizes data from a CSV file containing information about the executions of data integration processes. It aggregates and presents key metrics and insights, and includes features like K-Means clustering and interactive visualizations.

## Code Structure

1. Import necessary libraries including Streamlit, pandas, scikit-learn, matplotlib, dateutil, seaborn, and plotly.express.
2. Define the list of allowed execution types: 'exec_listener', 'exec_listener_bridge', 'exec_sched', 'sub_process'.
3. Load the CSV data from the specified file path using pandas' `read_csv` function. The 'executionId' column is set as the index column.
4. Convert the 'executionTime' column to a datetime format using the `pd.to_datetime` function and format string.
5. Create a new column 'Process Category' based on the presence of 'parentExecutionId', categorizing executions as 'Main Process' or 'Child Process'.
6. Aggregate the data by 'executionType' and calculate the mean of 'outboundDocumentCount', 'executionDuration', 'inboundDocumentSize', and 'outboundDocumentSize'.
7. Filter the aggregated data based on the allowed execution types.
8. Perform K-Means clustering on a subset of the data and add the 'cluster' column to the DataFrame.
9. Build the Streamlit app UI:
   - Display aggregated data in a subsection.
   - Display K-Means clustering results in a subsection.
   - Include interactive visualizations in the "Data Visualization" section:
     - Sidebar for filtering options (process category).
     - Interactive scatter plot of execution duration over time.
     - Interactive scatter plot of outbound document count vs. execution duration.
     - Interactive scatter plot of outbound document size vs. execution duration.
   - Display bar plot of execution type vs. outbound document count.
   - Display scatter plot of K-Means clusters based on execution duration and outbound document size.

## How to Run

1. Make sure you have Python and required libraries installed (Streamlit, pandas, scikit-learn, matplotlib, dateutil, seaborn, plotly.express).
2. Save the code to a file named `app.py`.
3. Open a terminal and navigate to the directory containing `app.py`.
4. Run the Streamlit app using the command:
   ```
   streamlit run app.py
   ```
5. A browser window should open displaying the interactive dashboard.

## Usage

- The dashboard provides insights into execution data and allows you to filter by process category.
- Interactive visualizations help you explore execution duration, document count, and document size.
- K-Means clustering results and bar plots offer further insights.

Feel free to customize the dashboard and add more features as needed!

---

Please note that this is a basic README example. You can modify and expand it based on your project's specific requirements and any additional information you want to provide to users.
