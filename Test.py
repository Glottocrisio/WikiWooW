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
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import spearmanr, kendalltau
from gplearn.genetic import SymbolicRegressor
# from mlxtend.frequent_patterns import apriori, association_rules
from mpl_toolkits.mplot3d import Axes3D
import shap

# def assrules(dataset):
#     df = pd.read_csv(dataset, sep=';')

#     df = df.iloc[:, [2, 5, 6, 7, 8, 9, 10, 12, 13]]
#     print(df.dtypes)
#     frequent_itemsets = apriori(df, min_support=0.5, use_colnames=False)
#     print(frequent_itemsets)

#     # Generating association rules
#     rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
#     print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

# assrules("updated_data.csv")


def testSimilarityIntercorrelation(dataset, selected_columns=None):
    df = pd.read_csv('C:\\Users\\Palma\\Desktop\\PHD\\WikiWooW\\'+str(dataset), sep=';')
    df = pd.read_csv(dataset, sep=';', header=0)
    selected_column = 'PalmaInterestingnessEnt1Ent2' 
    datarr = np.array(df['PalmaInterestingnessEnt1Ent2'])
    df['PalmaInterestingnessBool'] = df[selected_column].apply(lambda x: 1 if x > np.median(datarr) else 0)
    datarr = np.array(df['ground truth confidence values'])
    df['ground truth confidence values bool'] = df[selected_column].apply(lambda x: 1 if x > np.median(datarr) else 0)
    # Rank each column and store the ranks in new columns
    # for column in df.columns:
    #     df[column + '_rank'] = df[column].rank(method='min')
    # Convert all columns except the first two to integers
    #df.iloc[:, 2:] = df.iloc[:, 2:].astype(float).astype(int)

    print(df.head())
    selected_columns = df.iloc[:, [2, 10, 12, 13]] #df.iloc[:, [2, 5, 6, 7, 8, 9, 10, 13, 14, 15]]#, 16, 17, 18, 19, 20, 21, 22, 23, 24, 26, 27]] 

    correlations = selected_columns.corr(method='spearman')  # You can choose a different method if needed

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
    plt.title('Spearman\'s correlation Matrix Heatmap Alexander OUT')
    plt.xticks(np.arange(len(correlations)), correlations.columns, rotation=45)
    plt.yticks(np.arange(len(correlations)), correlations.columns)
    plt.show()

#visualize_correlations(testSimilarityIntercorrelation("updated_data.csv"))
# dataset = 'finaldataset_Alexander_light_annotated_out.tsv'
# df = pd.read_csv(dataset, sep=';')
# df = df.iloc[:, 2:]

#Random Forest Classifier  [2, 5, 6, 7, 8, 9, 11]

def rfc(dataset):
    df = pd.read_csv(dataset, sep=';')

    X = df.iloc[:, [2, 5, 6, 7, 8, 9, 10]].values 
    y = df.iloc[:, 12].values 
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

#rfc("updated_data.csv")

#RFR

def rfreg(dataset):
    df = pd.read_csv(dataset, sep=';')

    X = df.iloc[:, [2, 5, 6, 7, 8, 9, 10]].values 
    y = df.iloc[:, 12].values 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)

    # Get feature importances
    importances = model.feature_importances_
    #feature_names = X.c
    feature_importances = pd.DataFrame(importances, columns=['importance']).sort_values('importance', ascending=False)
    print(feature_importances)
    return feature_importances

#rfreg("updated_data.csv")


#SVM

def svm(dataset):
    df = pd.read_csv(dataset, sep=';')
    X = df.iloc[:, [2, 5, 6, 7, 8, 9, 10]].values  
    y = df.iloc[:, 12].values 
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

#svm("updated_data.csv")

#KNN

def knn(dataset):
    
    df = pd.read_csv(dataset, sep=';')

    X = df.iloc[:, [2, 5, 6, 7, 8, 9, 10]].values  
    y = df.iloc[:, 12].values 
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
    

#knn("updated_data.csv")

# PRINCIPAL COMPONENTS ANALYSIS

def pca(dataset):
    df = pd.read_csv(dataset, sep=';', encoding='ISO-8859-1')
    df_subset = df.iloc[:, 2:]
    scaler = StandardScaler()
    df_standardized = scaler.fit_transform(df_subset)
    pca = PCA()
    pca_result = pca.fit_transform(df_standardized)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # Export PCA results to CSV
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(pca_result.shape[1])])
    pca_df.to_csv('pca_result.csv', index=False)
    
    # Export explained variance ratio to CSV
    explained_variance_df = pd.DataFrame(explained_variance_ratio, columns=['Explained Variance Ratio'])
    explained_variance_df.to_csv('explained_variance_ratio.csv', index=False)
    
    print("Principal Components:")
    print(pd.DataFrame(pca.components_, columns=df_subset.columns))
    
    # Get the top 3 features contributing to each principal component
    top_features = []
    for pc in pca.components_[:3]:
        top_3 = df_subset.columns[np.argsort(np.abs(pc))[-3:]].tolist()
        top_features.append(", ".join(top_3))
    
    # Visualize in 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(pca_result[:, 0], pca_result[:, 1], pca_result[:, 2], c='r', marker='o')
    ax.set_title("3D PCA of WikiWooW Dataset")
    ax.set_xlabel(f"PC1 (CosineSimE1E2, InterestE1E2, DBRelE1E2)", fontsize=10)
    ax.set_ylabel(f"PC2 (PopE1, Ground_Truth_bin, PopSumE1E2)", fontsize=10)
    ax.set_zlabel(f"PC3 (Ground_Truth_bin, Ground_Truth_Float, PopEnt2)", fontsize=10)
    
    
    # Adjusting tick label font size
    ax.tick_params(axis='both', which='major', labelsize=8)
    ax.tick_params(axis='both', which='minor', labelsize=8)
    
    # Save the figure
    plt.savefig('pca_3d_plot.png', dpi=300, bbox_inches='tight')
    print("3D PCA plot saved as 'pca_3d_plot.png'")
    
    plt.show()

# Uncomment the following line to run the function
pca("updated_data.csv")
#pca("finaldataset.csv")

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

from sklearn.preprocessing import LabelEncoder


def calculate_shap_importance(input_file, output_file):
    # Read CSV file
    df = pd.read_csv(input_file, sep=';', encoding='ISO-8859-1')
    feature_columns = ['ClickstreamEnt1Ent2', 'PopularityEnt1', 'PopularityEnt2', 'PopularityDiff', 'PopularitySum', 
                       'CosineSimilarityEnt1Ent2', 'DBpediaSimilarityEnt1Ent2', 'DBpediaRelatednessEnt1Ent2', 
                       'PalmaInterestingnessEnt1Ent2']
    
    X = df[feature_columns]
    y = df['ground truth (threshold 0.8)']
    # Print some information about the data
    print(f"Number of samples: {len(X)}")
    print(f"Number of features: {len(feature_columns)}")
    print(f"Feature names: {feature_columns}")
    print(f"Target distribution: {y.value_counts(normalize=True)}")
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    explainer = shap.TreeExplainer(rf_model)
    shap_values = explainer.shap_values(X_test)
    # Print information about SHAP values
    print(f"SHAP values shape: {np.array(shap_values).shape}")
    # Plot SHAP values
    plt.figure(figsize=(12, 8))
    if isinstance(shap_values, list) and len(shap_values) > 1:
        # For binary classification
        shap.summary_plot(shap_values[1], X_test, plot_type="bar", show=False)
        feature_importance = dict(zip(X.columns, np.abs(shap_values[1]).mean(0)))
    else:
        # For regression or single-output classification
        shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
        feature_importance = dict(zip(X.columns, np.abs(shap_values).mean(0)))
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close()
    print(f"SHAP feature importance plot saved as {output_file}")
    # Ensure feature importance values are scalar
    feature_importance = {k: np.mean(v) if isinstance(v, np.ndarray) else v for k, v in feature_importance.items()}
    return feature_importance

input_file = "updated_data.csv"
output_file = "shap_feature_importance.png"
try:
    importance = calculate_shap_importance(input_file, output_file)
    # Print feature importance
    for feature, importance_value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
        print(f"{feature}: {importance_value}")
except Exception as e:
    print(f"An error occurred: {str(e)}")
    import traceback
    traceback.print_exc()
    print("Please check your input file and make sure it's formatted correctly.")

