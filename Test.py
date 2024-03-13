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
from sklearn.metrics import mean_squared_error, r2_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


def testSimilarityIntercorrelation(dataset, selected_columns=None):
    df = pd.read_csv('C:\\Users\\Palma\\Desktop\\PHD\\WikiWooW\\'+str(dataset), sep=';')
    df = pd.read_csv(dataset, sep=';', header=0)
    selected_column = 'PalmaInterestingnessEnt1Ent2'  
    df['PalmaInterestingnessBool'] = df[selected_column].apply(lambda x: 1 if x > 5 else 0)
    
    if selected_columns is None:
        selected_columns = df.select_dtypes(include=['number']).columns

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

dataset = 'finaldataset_Alexander_light_annotated_out.tsv'
df = pd.read_csv(dataset, sep=';')
df = df.iloc[:, 2:]

#Random Forest Classifier  [2, 5, 6, 7, 8, 9, 11]

def rfc(dataset):
    df = pd.read_csv(dataset, sep=';')

    X = df.iloc[:, [2, 5, 6, 7, 8, 9]].values 
    y = df.iloc[:, -1].values 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf_classifier = RandomForestClassifier(n_estimators=100)  
    rf_classifier.fit(X_train, y_train)
    y_pred = rf_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy of Random Forest classifier: {accuracy:.2f}")
    print(f"precision of Random Forest classifier: {precision:.2f}")
    print(f"recall of Random Forest classifier: {recall:.2f}")
    print(f"f1 of Random Forest classifier: {f1:.2f}")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=rf_classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf_classifier.classes_)
    disp.plot()
    plt.show()


#SVM

def svm(dataset):
    df = pd.read_csv(dataset, sep=';')
    X = df.iloc[:, [2, 5, 6, 7, 8, 9]].values  
    y = df.iloc[:, -1].values 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    svm_classifier = SVC(kernel='linear')  
    svm_classifier.fit(X_train, y_train)

    y_pred = svm_classifier.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy of SVM classifier: {accuracy:.2f}")
    print(f"precision of SVM classifier: {precision:.2f}")
    print(f"recall of SVM classifier: {recall:.2f}")
    print(f"f1 of SVM classifier: {f1:.2f}")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=svm_classifier.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=svm_classifier.classes_)
    disp.plot()
    plt.show()


#KNN

def knn(dataset):
    
    df = pd.read_csv(dataset, sep=';')

    X = df.iloc[:, [2, 5, 6, 7, 8, 9]].values  
    y = df.iloc[:, -1].values 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    k = 5  
    knn = KNeighborsClassifier(n_neighbors=k)

    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy of K-NN classifier with k={k}: {accuracy:.2f}")
    print(f"precision of K-NN classifier: {precision:.2f}")
    print(f"recall of K-NN classifier: {recall:.2f}")
    print(f"f1 of K-NN classifier: {f1:.2f}")
    print(classification_report(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=knn.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=knn.classes_)
    disp.plot()
    plt.show()
    

# PRINCIPAL COMPONENTS ANALYSIS

def pca(dataset):
    df = pd.read_csv(dataset, sep=';')
    df_subset = df.iloc[:, 2:]
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df_subset)
    pca = PCA()
    pca_result = pca.fit_transform(df_standardized)
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Principal Components:")
    print(pd.DataFrame(pca.components_, columns=df_subset.columns))

##ISOLATION FOREST

def isoforest(dataset,x,y):
    df = pd.read_csv(dataset, sep=';')
    iso_forest = IsolationForest(contamination=0.05)  
    outliers = iso_forest.fit_predict(df)
    df['is_outlier'] = outliers
    df.plot.scatter(x=x, y=y, c='is_outlier', cmap='viridis')


# ##CLUSTER ANALYSIS

def cluster(dataset, x, y):
    num_clusters = 2
    df = pd.read_csv(dataset, sep=';')
    kmeans = KMeans(n_clusters=num_clusters)
    df['cluster'] = kmeans.fit_predict(df)

    plt.scatter(df.iloc[:, 0], df.iloc[:, 1], c=df['cluster'], cmap='viridis')
    plt.title('Cluster Analysis')
    plt.xlabel(x)
    plt.ylabel(y)
    plt.show()
    

def regression(dataset):
    df = pd.read_csv(dataset, sep=';')
    X = df.iloc[:, [2, 9]].values  
    y = df.iloc[:, -1].values  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("R-squared:", r2)
   
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


def MultnaiveBayes(dataset):
    nb = MultinomialNB()
    df = pd.read_csv(dataset, sep=';')
    X = df.iloc[:, [2, 5, 6, 7, 8, 9]].values 
    y = df.iloc[:, -1].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    nb.fit(X_train, y_train)
    nb_predict = nb.predict(X_test)

    cm = confusion_matrix(y_test, nb_predict, labels=nb.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=nb.classes_)
    disp.plot()
    plt.show()
    print('\n')
    print(classification_report(y_test, nb_predict))
