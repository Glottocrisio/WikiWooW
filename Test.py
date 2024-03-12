#This module will implement the experiment runt on the WikiWooW dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def testSimilarityIntercorrelation(dataset, selected_columns=None):
    # Read the TSV file into a DataFrame
    df = pd.read_csv(dataset, sep=';')
    df = pd.read_csv(dataset, sep=';', header=0)
    
    # If selected_columns is not provided, analyze all numeric columns
    if selected_columns is None:
        selected_columns = df.select_dtypes(include=['number']).columns

    # Calculate correlations for selected columns
    correlations = df[selected_columns].corr(method='pearson')  # You can choose a different method if needed

    return correlations

def visualize_correlations(correlations):
    plt.figure(figsize=(10, 8))
    
    fig, ax = plt.subplots()
    im =ax.imshow(correlations, cmap='coolwarm', interpolation='nearest', aspect='auto')
    for i in range(len(correlations)):
        for j in range(len(correlations)):
            text = ax.text(j, i, f'{correlations.iloc[i, j]:.2f}',
                           ha='center', va='center', color='black')
    plt.colorbar(im, label='Correlation Coefficient')
    plt.title('Correlation Matrix Heatmap with Annotations')
    plt.xticks(np.arange(len(correlations)), correlations.columns, rotation=45)
    plt.yticks(np.arange(len(correlations)), correlations.columns)
    plt.show()

dataset = 'temp_datasetintfinalclean.tsv'
df = pd.read_csv(dataset, sep=';')
df = df.iloc[:, 2:]

# PRINCIPAL COMPONENTS ANALYSIS

def pca(dataset):
    df_subset = df.iloc[:, 2:]
    # Standardize the data
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df_subset)
    # Apply PCA
    pca = PCA()
    pca_result = pca.fit_transform(df_standardized)
    # Access explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_
    # Print the first few principal components
    print("Principal Components:")
    print(pd.DataFrame(pca.components_, columns=df_subset.columns))

##ISOLATION FOREST

def isoforest(dataset,x,y):
    iso_forest = IsolationForest(contamination=0.05)  # Adjust contamination based on your data
    # Fit the model and predict outliers
    outliers = iso_forest.fit_predict(df)
    # Add the outlier labels to the DataFrame
    df['is_outlier'] = outliers
    # Visualize outliers
    df.plot.scatter(x=x, y=y, c='is_outlier', cmap='viridis')


# ##CLUSTER ANALYSIS

def cluster(dataset, x, y):
    num_clusters = 2

    # Fit KMeans model
    kmeans = KMeans(n_clusters=num_clusters)
    df['cluster'] = kmeans.fit_predict(df)

    # Visualize clusters
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['cluster'], cmap='viridis')
    plt.title('Cluster Analysis')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    

def regression(dataset):
    # Assuming X contains all columns except the first two, and y is the third column
    X = df.values
    y = df.iloc[:, 8].values  # Adjust the column index based on your data
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create a linear regression model
    model = LinearRegression()
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)
    # Evaluate the model
    mse = mean_squared_error(y_test, y_pred)
    print(f'Mean Squared Error: {mse}')
   
    # Visualize the regression line
    plt.scatter(y_test, y_pred, color='blue', label='Actual')
    #plt.scatter(X_test, y_pred, color='red', label='Predicted')
    plt.xlabel('Independent Variable')
    plt.ylabel('Dependent Variable')
    plt.title('Regression Analysis')
    plt.legend()
    plt.show()
    #print(classification_report(y_test, y_pred))
    #print(classification_report(y_test, y_pred)) 


def MultnaiveBayes():
    nb = MultinomialNB()
    df = pd.read_csv(dataset, sep=';')
    df =df.iloc[:, 2:]
    X = df.values
    y = df.iloc[:, 8].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb.fit(X_train, y_train)
    #Testing our model
    nb_predict = nb.predict(X_test)

    #Creating the confusion matrix
    
    cm = confusion_matrix(y_test, nb_predict, labels=nb.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb.classes_)
    disp.plot()
    plt.show()
    print('\n')
    print(classification_report(y_test, nb_predict))