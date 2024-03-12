import tools as t
from rdflib import Graph, URIRef, Literal, RDF, RDFS
import requests
import clkStreamDataset as clks
import Test as tst
import json
import pandas as pd
import matplotlib.pyplot as plt



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
i=1
while i<=4:
    clks.addPopularityclkstrdataset("output_partition_alex"+str(i)+".tsv")
    # OR
    # i=1
    # while i < 12:
    #     clks.addPopularityclkstrdataset("clkstrdataset"+str(i)+".tsv")
    #     i+=1


    ##4. Append corpus-based and knowledge-based similarity values to the dataset

    clks.addKSimilarityclkstrdataset("popoutput_partition_alex"+str(i)+".tsv")

    ##5. Append P-Interestingness value to the dataset
    dataset = "simpopoutput_partition_alex"+str(i)+".tsv"
    t.datasetcleansing(dataset)
    clks.addInterestingnessclkstrdataset('clean'+dataset)
    i+=1

##6. (Optional) add features vectors to the dataset
##clks.addFeaturesclkstrdataset("temp_datasetpop.tsv")

##7. Clean up the final dataset (Remove rows containing at least a zero to ease out 
##later calculations)
#dataset = 'temp_datasetintfinal.tsv'
#t.datasetcleansing(dataset)


## After building up the dataset, all kinds of experiments are possible with it
## Among them we consider these:

##1 Check for correlation among all paramenters belonging to the same category 
##(Popularity, Similarity)
##a) Popularity
# selected_columns = ['PopularityEnt1','PopularityEnt2', 'PopularityDiff', 'PopularitySum'] 
# correlations = tst.testSimilarityIntercorrelation('temp_datasetintfinalclean.tsv', selected_columns)
# tst.visualize_correlations(correlations)

# ##b) Similarity

# selected_columns = ['CosineSimilarityEnt1Ent2', 'DBpediaSimilarityEnt1Ent2', 'DBpediaRelatednessEnt1Ent2'] 
# correlations = tst.testSimilarityIntercorrelation('temp_datasetintfinalclean.tsv', selected_columns)
# tst.visualize_correlations(correlations)

# ##2 Check for cross-category correlations, to see how these are intertwined:

# correlations = tst.testSimilarityIntercorrelation('temp_datasetintfinalclean.tsv')
# tst.visualize_correlations(correlations)

# ##3 Based on that considerations and on the law extrapolated in a qualitative way 
# ##(just refer, in this case, to the paper you wrote for text to story)


# selected_columns = ['ClickstreamEnt1Ent2', 'PopularitySum'] 
# correlations = tst.testSimilarityIntercorrelation('temp_datasetintfinalclean.tsv', selected_columns)
# tst.visualize_correlations(correlations)

# selected_columns = ['ClickstreamEnt1Ent2', 'CosineSimilarityEnt1Ent2'] 
# correlations = tst.testSimilarityIntercorrelation('temp_datasetintfinalclean.tsv', selected_columns)
# tst.visualize_correlations(correlations)

# selected_columns = ['ClickstreamEnt1Ent2', 'DBpediaRelatednessEnt1Ent2'] 
# correlations = tst.testSimilarityIntercorrelation('temp_datasetintfinalclean.tsv', selected_columns)
# tst.visualize_correlations(correlations)

# selected_columns = ['CosineSimilarityEnt1Ent2', 'DBpediaSimilarityEnt1Ent2'] 
# correlations = tst.testSimilarityIntercorrelation('temp_datasetintfinalclean.tsv', selected_columns)
# tst.visualize_correlations(correlations)

# tst.pca('temp_datasetintfinalclean.tsv')
# tst.isoforest('temp_datasetintfinalclean.tsv', 'PalmaInterestingnessEnt1Ent2', 'ClickstreamEnt1Ent2')
# tst.cluster('temp_datasetintfinalclean.tsv', 'CosineSimilarityEnt1Ent2', 'DBpediaSimilarityEnt1Ent2')

# tst.regression('temp_datasetintfinalclean.tsv')


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