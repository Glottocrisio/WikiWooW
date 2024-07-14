import tools as t
from rdflib import Graph, URIRef, Literal, RDF, RDFS
import requests
import clkStreamDataset as clks
import Test as tst
import json
import pandas as pd
import matplotlib.pyplot as plt

#t.addgroundtruth("finaldataset.csv", "ground_truth_interestingness.json")

##This module implements the principal pipeline derived combining the other modules

## The first phase is the costruction of the dataset

# 1. Clickstream data-dump partitioning

# t.file_partitioning('enwiki_20180420_100d.txt', 50)
# t.file_partitioning('enwiki_20180420_nolg_100d.txt', 50)

## For every file, execute the following:

## Preprocessing

# Remove third column, where 'other' and 'link' are written

# i=1
# while i < 12:
#     clks.preprocessclkstrdataset("clickstream-enwiki-2023-11_"+str(i)+".tsv", "clkstrdataset"+str(i)+".tsv")
#     i+=1

#1.1 Search through all clickstream dumps to retrieve all couplets starting and ending with a given entity 
#(in- and outgoing links)

# entity = input()
# wikiPageWikiLinkFile = "wikiPageWikiLink"+str(entity)+".tsv"
# i=1
# while i < 10:
#     j = 1
#     t.file_partitioning("C:\\Users\\Palma\\Desktop\\PHD\\ISAAKbackup\\ISAAKx\\clkstrdataset"+str(i)+".tsv", 100)
#     while j<100:
#         clks.findwikiPageWikiLink("C:\\Users\\Palma\\Desktop\\PHD\\WikiWooW\\output_partition_"+str(j)+".tsv", wikiPageWikiLinkFile, entity)
#         j += 1
#     i+=1


## 2. Preprocess the dataset to align it with the .tsv format

#clks.preprocessclkstrdataset("clickstream-enwiki-2023-11_5.tsv", "mockupfinal.tsv")
#t.file_partitioning("wikiPageWikiLinkAlexander_the_Great.tsv", 21)
##3. Append the popularity values to the dataset 
# dataset = "simpopoutput_partition_alex12.tsv"
# t.datasetcleansing(dataset)
# clks.addInterestingnessclkstrdataset('clean'+dataset)
# i=1
# while i<=4:
#     clks.addPopularityclkstrdataset("output_partition_alex"+str(i)+".tsv")
#     # OR
#     # i=1
#     # while i < 12:
#     #     clks.addPopularityclkstrdataset("clkstrdataset"+str(i)+".tsv")
#     #     i+=1


#     ##4. Append corpus-based and knowledge-based similarity values to the dataset

#     clks.addKSimilarityclkstrdataset("popoutput_partition_alex"+str(i)+".tsv")

#     ##5. Append P-Interestingness value to the dataset
#     dataset = "simpopoutput_partition_alex"+str(i)+".tsv"
#     t.datasetcleansing(dataset)
#     clks.addInterestingnessclkstrdataset('clean'+dataset)
#     i+=1

##6. (Optional) add features vectors to the dataset
##clks.addFeaturesclkstrdataset("temp_datasetpop.tsv")

##7. Clean up the final dataset (Remove rows containing at least a zero to ease out 
##later calculations)
#dataset = 'temp_datasetintfinal.tsv'
#t.datasetcleansing(dataset)


## After building up the dataset, all kinds of experiments are possible with it
## Among them we consider these:

#Features comparison and analysis

# tst.pca('finaldataset_Alexander_light_annotated_out.tsv')
# tst.pca('finaldataset_Alexander_light_annotated_in.tsv')
# tst.pca('temp_datasetfinalAnubis.tsv')


#tst.regression('finaldataset_Alexander_light_annotated_in.tsv')
#tst.regression('finaldataset_Alexander_light_annotated_out.tsv')
#tst.regression('temp_datasetfinalAnubis.tsv')


# selected_columns = ['PalmaInterestingnessEnt1Ent2', 'ClickstreamEnt1Ent2', 'ground truth (threshold 0.79)', 'ground truth confidence values'] 
# correlations = tst.testSimilarityIntercorrelation('finaldataset_Alexander_light_annotated_in.csv', selected_columns)
# tst.visualize_correlations(correlations)

selected_columns = ['PalmaInterestingnessEnt1Ent2', 'ClickstreamEnt1Ent2', 'ground truth (threshold 0.79)', 'ground truth confidence values'] 
correlations = tst.testSimilarityIntercorrelation('finaldataset_Alexander_light_annotated_out.csv', selected_columns)
tst.visualize_correlations(correlations)



selected_columns = ['ClickstreamEnt1Ent2','PopularityEnt1','PopularityEnt2','PopularityDiff','PopularitySum','CosineSimilarityEnt1Ent2','DBpediaSimilarityEnt1Ent2','DBpediaRelatednessEnt1Ent2','PalmaInterestingnessEnt1Ent2','Int_Hum_Eval'] 
correlations = tst.testSimilarityIntercorrelation('temp_datasetfinalAnubis.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['ClickstreamEnt1Ent2','PopularityEnt1','PopularityEnt2','PopularityDiff','PopularitySum','CosineSimilarityEnt1Ent2','DBpediaSimilarityEnt1Ent2','DBpediaRelatednessEnt1Ent2','PalmaInterestingnessEnt1Ent2','Int_Hum_Eval'] 
correlations = tst.testSimilarityIntercorrelation('finaldataset_Alexander_light_annotated_inout.tsv', selected_columns)
tst.visualize_correlations(correlations)





tst.svm('finaldataset_Alexander_light_annotated_in.tsv')
tst.svm('finaldataset_Alexander_light_annotated_out.tsv')

tst.MultnaiveBayes('finaldataset_Alexander_light_annotated_in.tsv')
tst.MultnaiveBayes('finaldataset_Alexander_light_annotated_out.tsv')

tst.knn('finaldataset_Alexander_light_annotated_in.tsv')
tst.knn('finaldataset_Alexander_light_annotated_out.tsv')

tst.rfc('finaldataset_Alexander_light_annotated_in.tsv')
tst.rfc('finaldataset_Alexander_light_annotated_out.tsv')





selected_columns = ['ClickstreamEnt1Ent2', 'PopularitySum'] 
correlations = tst.testSimilarityIntercorrelation('finaldataset_Alexander.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['ClickstreamEnt1Ent2', 'PopularitySum'] 
correlations = tst.testSimilarityIntercorrelation('temp_datasetfinalAnubis.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['ClickstreamEnt1Ent2', 'CosineSimilarityEnt1Ent2'] 
correlations = tst.testSimilarityIntercorrelation('finaldataset_Alexander.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['ClickstreamEnt1Ent2', 'CosineSimilarityEnt1Ent2'] 
correlations = tst.testSimilarityIntercorrelation('temp_datasetfinalAnubis.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['CosineSimilarityEnt1Ent2', 'DBpediaRelatednessEnt1Ent2'] 
correlations = tst.testSimilarityIntercorrelation('finaldataset_Alexander.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['CosineSimilarityEnt1Ent2', 'DBpediaRelatednessEnt1Ent2'] 
correlations = tst.testSimilarityIntercorrelation('temp_datasetfinalAnubis.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['ClickstreamEnt1Ent2', 'DBpediaRelatednessEnt1Ent2'] 
correlations = tst.testSimilarityIntercorrelation('finaldataset_Alexander.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['ClickstreamEnt1Ent2', 'DBpediaRelatednessEnt1Ent2'] 
correlations = tst.testSimilarityIntercorrelation('temp_datasetfinalAnubis.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['ClickstreamEnt1Ent2', 'Int_Hum_Eval'] 
correlations = tst.testSimilarityIntercorrelation('finaldataset_Alexander_light_annotated_in.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['ClickstreamEnt1Ent2', 'Int_Hum_Eval']
correlations = tst.testSimilarityIntercorrelation('temp_datasetfinalAnubis.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['ClickstreamEnt1Ent2', 'PalmaInterestingnessEnt1Ent2'] 
correlations = tst.testSimilarityIntercorrelation('finaldataset_Alexander.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['ClickstreamEnt1Ent2', 'PalmaInterestingnessEnt1Ent2'] 
correlations = tst.testSimilarityIntercorrelation('temp_datasetfinalAnubis.tsv', selected_columns)
tst.visualize_correlations(correlations)

# tst.pca('finaldataset_Alexander_light_annotated_out.tsv')
# tst.isoforest('temp_datasetintfinalclean.tsv', 'PalmaInterestingnessEnt1Ent2', 'ClickstreamEnt1Ent2')
# tst.cluster('temp_datasetintfinalclean.tsv', 'CosineSimilarityEnt1Ent2', 'DBpediaSimilarityEnt1Ent2')


##Comparison of interestingness between in- and outgoing links dataset


df = pd.read_csv('finaldataset_Alexander_light_annotated_out.tsv', sep=';')

selected_column = 'PalmaInterestingnessEnt1Ent2'  
column_name = 'PalmaInterestingnessEnt1Ent2' 
average_value = df[column_name].mean()

print(f"The average value of column '{column_name}' is: {average_value}")

df['PalmaInterestingnessBool'] = df[selected_column].apply(lambda x: 1 if x > average_value else 0)

column_name = 'PalmaInterestingnessBool'  
average_value = df[column_name].mean()

print(f"The average value of column '{column_name}' is: {average_value}")

column_name = 'Int_Hum_Eval'  
average_value = df[column_name].mean()

print(f"The average value of column '{column_name}' is: {average_value}")

df = pd.read_csv('finaldataset_Alexander_light_annotated_in.tsv', sep=';')

column_name = 'PalmaInterestingnessEnt1Ent2' 
average_value = df[column_name].mean()
df['PalmaInterestingnessBool'] = df[selected_column].apply(lambda x: 1 if x > average_value else 0)

print(f"The average value of column '{column_name}' is: {average_value}")

column_name = 'PalmaInterestingnessBool'  
average_value = df[column_name].mean()
print(f"The average value of column '{column_name}' is: {average_value}")


column_name = 'Int_Hum_Eval'  
average_value = df[column_name].mean()
print(f"The average value of column '{column_name}' is: {average_value}")

df = pd.read_csv('temp_datasetfinalAnubis.tsv', sep=';')

column_name = 'PalmaInterestingnessEnt1Ent2'  
average_value = df[column_name].mean()
df['PalmaInterestingnessBool'] = df[selected_column].apply(lambda x: 1 if x > average_value else 0)

print(f"The average value of column '{column_name}' is: {average_value}")

column_name = 'PalmaInterestingnessBool'  
average_value = df[column_name].mean()
print(f"The average value of column '{column_name}' is: {average_value}")


column_name = 'Int_Hum_Eval'  
average_value = df[column_name].mean()
print(f"The average value of column '{column_name}' is: {average_value}")
# # Plot the data
# plt.figure(figsize=(8, 6))
# plt.scatter(x_values, y_values, color='blue', alpha=0.5)  # Scatter plot
# plt.title('Plot of Column X vs Column Y')
# plt.xlabel('Column X')
# plt.ylabel('Column Y')
# plt.grid(True)
# plt.show()

selected_columns = ['PalmaInterestingnessBool', 'PalmaInterestingnessEnt1Ent2', 'Int_Hum_Eval'] 
correlations = tst.testSimilarityIntercorrelation('temp_datasetfinalAnubis.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['PalmaInterestingnessBool','PalmaInterestingnessEnt1Ent2', 'Int_Hum_Eval'] 
correlations = tst.testSimilarityIntercorrelation('finaldataset_Alexander_light_annotated_in.tsv', selected_columns)
tst.visualize_correlations(correlations)

selected_columns = ['PalmaInterestingnessBool','PalmaInterestingnessEnt1Ent2', 'Int_Hum_Eval'] 
correlations = tst.testSimilarityIntercorrelation('finaldataset_Alexander_light_annotated_out.tsv', selected_columns)
tst.visualize_correlations(correlations)

##3 Based on that considerations and on the law extrapolated in a qualitative way 
##(just refer, in this case, to the paper you wrote for text to story)


#Read TSV file into a DataFrame
# df = pd.read_csv('temp_datasetintfinalclean.tsv', sep=';')

# # Display the first few rows of the DataFrame
# print("DataFrame Head:")
# print(df.head())

# # Create a bar chart using a sample of the data
# sample_data = df.head(30)  # Adjust the number of rows to display
# sample_data.plot(kind='bar', x='ClickstreamEnt1Ent2', y='PopularityDiff', legend=False)
# plt.xlabel('X-axis Label')  # Replace with your column name
# plt.ylabel('Y-axis Label')  # Replace with your column name
# plt.title('Clickstream and popularity correlation')
# plt.show()

# dataset = 'temp_datasetintfinalclean.tsv'
# df = pd.read_csv(dataset, sep=';')
# df = df.iloc[:, 2:]


# # Extract the two numeric columns
# x_values = df['ClickstreamEnt1Ent2']
# y_values = df['CosineSimilarityEnt1Ent2']

# # Create a scatter plot
# plt.scatter(x_values, y_values, marker='o', color='blue')
# plt.title('Scatter Plot of x and y')
# plt.xlabel('x-axis label')
# plt.ylabel('y-axis label')
# plt.show()