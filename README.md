# stock-clostering-demo
Machine Learning demo using scaling, encoding, K-means clustering and PCA with energy stock data.

The goal of this exercise is to standardize the data points that include "MeanOpen", "MeanHigh", "MeanLow", "MeanClose", "MeanVolume", "AnnualReturn", "AnnualVariance", use K-means clustering to segment data to potentially gain insight and then determine if reducing the data dimensionality with PCA will achieve similar amd possibly more insightful results.

Full code available in "stock_clustering.ipynb" notebook.

```
# After importing dependcies create DataFrame.
```
![Screenshot 2023-12-18 at 2 22 44 PM](https://github.com/samuelhfish/stock-clostering-demo/assets/125224990/62ceae61-917a-469a-a58e-9cd3d5a433d8)

```python
# Scale price data, return, and variance values
stock_data_scaled = StandardScaler().fit_transform(
    df_stocks[["MeanOpen", "MeanHigh", "MeanLow", "MeanClose", "MeanVolume", "AnnualReturn", "AnnualVariance"]]
)
```
```python
# Create a new DataFrame with the scaled data
df_stocks_scaled = pd.DataFrame(
    stock_data_scaled,
    columns=["MeanOpen", "MeanHigh", "MeanLow", "MeanClose", "MeanVolume", "AnnualReturn", "AnnualVariance"]
)

# Copy the tickers names from the original data
df_stocks_scaled["Ticker"] = df_stocks.index

# Set the Ticker column as index
df_stocks_scaled = df_stocks_scaled.set_index("Ticker")

# Display sample data
df_stocks_scaled.head()
```
![Screenshot 2023-12-18 at 2 26 41 PM](https://github.com/samuelhfish/stock-clostering-demo/assets/125224990/601e41d1-a0d6-43ac-ac39-d4288ca41df2)

```python
# Encode the "EnergyType" column to variables to categorize oil versus non-oil firms.

oil_dummies = pd.get_dummies(df_stocks["EnergyType"]).astype('int')
oil_dummies.head()

# Concatenate the "EnergyType" variables with the scaled data DataFrame.
df_stocks_scaled = pd.concat([df_stocks_scaled, oil_dummies], axis=1)

# Preapare data for modeling with K-means by removing extra column.
df_stocks_scaled = df_stocks_scaled.iloc[:,:-1]
```
![Screenshot 2023-12-18 at 2 32 23 PM](https://github.com/samuelhfish/stock-clostering-demo/assets/125224990/f23e3603-912c-46e5-929b-2dd8f40d6bf9)

```python
# Initialize the K-Means model with n_clusters=3
model = KMeans(n_clusters=3)

# Fit the model for the df_stocks_scaled DataFrame
model.fit(df_stocks_scaled)
```
```
Results:

KMeans(n_clusters=3)
```
```python
# Predict the clusters and then create a new DataFrame with the predicted clusters.
# Predict the model segments (clusters)
stock_clusters = model.predict(df_stocks_scaled)

# Create a new column in the DataFrame with the predicted clusters
df_stocks_scaled["StockCluster"] = stock_clusters

# Review the DataFrame
df_stocks_scaled.head()
```
![Screenshot 2023-12-18 at 2 35 01 PM](https://github.com/samuelhfish/stock-clostering-demo/assets/125224990/94841992-f940-4681-8e42-8e5e9e4e4be5)

```python
# Create a scatter plot with x="AnnualVariance' and y="AnnualReturn"
df_stocks_scaled.hvplot.scatter(
    x="MeanVolume",
    y="AnnualReturn",
    by="StockCluster",
    hover_cols = ["Ticker"], 
    title = "Scatter Plot by Stock Segment - k=3"
)
```
![Screenshot 2023-12-18 at 2 35 57 PM](https://github.com/samuelhfish/stock-clostering-demo/assets/125224990/48dc376f-b47d-4c4f-9e71-d03bd72f6252)

```python
# Remove initial clustering result from df
df_stocks_scaled = df_stocks_scaled.iloc[:,:-1]

# Create the PCA model instance and reduce numner of components so n_components=2
pca = PCA(n_components=2)

# Fit the df_stocks_scaled data to the PCA
stocks_pca_data = pca.fit_transform(df_stocks_scaled)

# Review the first five rose of the PCA data
# using bracket notation ([0:5])
stocks_pca_data[:5]
```
```
array([[-2.01541918,  0.46518931],
       [-1.62885632, -1.40685588],
       [ 1.85394351,  1.39068316],
       [-2.2941301 ,  1.95995804],
       [-3.04963345,  2.50345178]])
```
```python
# Calculate the explained variance
pca.explained_variance_ratio_
```
```
array([0.64467721, 0.1714023 ])
```
```python
# Create a newDataFrame with the PCA data
df_stocks_pca = pd.DataFrame(stocks_pca_data, columns=["PC1", "PC2"])

# Copy the tickers names from the original data
df_stocks_pca["Ticker"] = df_stocks.index

# Set the Ticker column as index
df_stocks_pca = df_stocks_pca.set_index("Ticker")

# Review the DataFrame
df_stocks_pca.head()
```
![Screenshot 2023-12-18 at 2 39 30 PM](https://github.com/samuelhfish/stock-clostering-demo/assets/125224990/33eba74c-4ec8-4886-95b9-3fdfce10072e)

```
Replicated K-means algorithm with PCA Data. Full code available in "stock_clustering.ipynb" notebook above.

Results:
```
![Screenshot 2023-12-18 at 2 41 15 PM](https://github.com/samuelhfish/stock-clostering-demo/assets/125224990/51a63b88-b560-4589-b462-d81eb5fa38ee)

![Screenshot 2023-12-18 at 2 41 53 PM](https://github.com/samuelhfish/stock-clostering-demo/assets/125224990/dda96f36-593a-4dfb-b4ad-0b76d998c9d6)

```python
# Combine results with original data
df_all = pd.concat([df_stocks, df_stocks_pca_predictions],axis=1)
df_all
```
![Screenshot 2023-12-18 at 2 43 35 PM](https://github.com/samuelhfish/stock-clostering-demo/assets/125224990/c62a3ebc-6176-4bf4-8de2-30d3e603c203)

```python
for cluster in df_all.StockCluster.unique():
    print(cluster)
    print(df_all[df_all.StockCluster == cluster]["CompanyName"])
```
```
1
Ticker
ARX               ARC Resources Ltd.
CVE              Cenovus Energy Inc.
CPG      Crescent Point Energy Corp.
MEG                 MEG Energy Corp.
VII    Seven Generations Energy Ltd.
WCP          Whitecap Resources Inc.
Name: CompanyName, dtype: object
2
Ticker
CCO         Cameco Corporation
ERF       Enerplus Corporation
GEI         Gibson Energy Inc.
HSE          Husky Energy Inc.
IPL        Inter Pipeline Ltd.
KEY               Keyera Corp.
MTL          Mullen Group Ltd.
PXT       Parex Resources Inc.
PKI       Parkland Corporation
PSK    PrairieSky Royalty Ltd.
Name: CompanyName, dtype: object
0
Ticker
CNQ    Canadian Natural Resources Limited
...
TRP                 TC Energy Corporation
TOU                  Tourmaline Oil Corp.
VET                 Vermilion Energy Inc.
Name: CompanyName, dtype: object
```

```python
pca.components_
```
```
array([[ 0.45909865,  0.45901713,  0.45920688,  0.45914138,  0.00921683,
         0.12452821, -0.3752216 ,  0.02159781],
       [ 0.07940306,  0.07958659,  0.078522  ,  0.07863812,  0.78850296,
        -0.51967858,  0.24270737,  0.15589053]])
```
```python
# Use elbow method to find best value for k with pca data.

# Create a list with the number of k-values to try
# Use a range from 1 to 11
k = list(range(1, 11))

# Create an empy list to store the inertia values
inertia = []

# Create a for loop to compute the inertia with each possible value of k
# Inside the loop:
# 1. Create a KMeans model using the loop counter for the n_clusters
# 2. Fit the model to the data using `df_stocks_pca`
# 3. Append the model.inertia_ to the inertia list
for i in k:
    model = KMeans(n_clusters=i, random_state=0)
    model.fit(df_stocks_pca)
    inertia.append(model.inertia_)
```
```python
# Create a dictionary with the data to plot the Elbow curve
elbow_data_pca = {
    "k": k,
    "inertia": inertia
}

# Create a DataFrame with the data to plot the Elbow curve
df_elbow_pca = pd.DataFrame(elbow_data_pca)
```
```python
# Plot a line chart with all the inertia values computed with 
# the different values of k to visually identify the optimal value for k.
elbow_plot_pca = df_elbow_pca.hvplot.line(x="k", y="inertia", title="Elbow Curve Using PCA Data", xticks=k)
elbow_plot_pca
```
![Screenshot 2023-12-18 at 2 48 06 PM](https://github.com/samuelhfish/stock-clostering-demo/assets/125224990/1872971f-294b-4828-b93b-7544c62fbf4e)

### Analysis:
Based on the PCA results and the elbow curve mthod we are able to get similar results with fewer features and also see that 3 is still the best value for k. 