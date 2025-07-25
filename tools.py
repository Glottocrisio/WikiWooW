import requests
from bs4 import BeautifulSoup
from collections import deque
from SPARQLWrapper import SPARQLWrapper, JSON
import re
import urllib.parse, urllib.request, json
from itertools import combinations
from rdflib import Graph, Namespace, Literal, URIRef, RDF, RDFS, BNode
import json
import csv
import pandas as pd
import os
import matplotlib.pyplot as plt
from rdflib.namespace import XSD
import json
import morph_kgc
from sklearn.feature_extraction import text
import nltk
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk import FreqDist
from nltk.tokenize import word_tokenize  #NLP library
import ast
#from mlxtend. import apriori, association_rules

stop_words = text.ENGLISH_STOP_WORDS
#print(stop_words)
stop_words = list(stop_words)


# # Finding frequent itemsets
# frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
# print(frequent_itemsets)

# # Generating association rules
# rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
# print(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']])

#to be enhanced after results are finalized
def addgroundtruth(input_file, ground_truth_file):
    #take input file
    original_data = pd.read_csv(input_file)
    mturk_labels = []
    mturk_conf_values = []
    # For each line, extract the confidence value. If above 0.85, assign value 1, 0 otherwise
    with open(ground_truth_file, 'r') as json_file: 
        for line in json_file:
            data = json.loads(line)
            label = data['interestingness-detector-metadata']['confidence']
            mturk_conf_values.append(float(label))
            if float(label) > 0.85:
                mturk_labels.append(1)
            else:
                mturk_labels.append(0)

    original_data['ground truth (threshold 0.80)'] = mturk_labels
    original_data['ground truth confidence values'] = mturk_conf_values
    original_data.to_csv("updated_data.csv", index=False)

def file_partitioning(file_path, num_partitions):

    df = pd.read_csv(file_path, sep='\t')
    rows_per_partition = len(df) // num_partitions
    partitions = [df.iloc[i:i+rows_per_partition] for i in range(0, len(df), rows_per_partition)]

    for i, partition in enumerate(partitions):
        output_file_path = f'output_partition_alex{i+1}.tsv'
        partition.to_csv(output_file_path, sep='\t', index=False)

    print("Partitioning complete.")
    
file_partitioning('C:\\Users\\Palma\\Desktop\\PHD\\ISAAKbackup\\ISAAKx\\clickstream-hrwiki-2025-03.tsv', 5)

# Remove Stop Words form the text
def cleansing(text):
    text = text.lower()
    word_list = word_tokenize(text)
    word_list = [word for word in word_list if len(word) > 2] #words shorter than three characters are excluded from the dataset
    #word_list = [remove_punctuation(word) for word in word_list]  #punctuation is removed
    text = ' '.join(word_list)
    fdist = FreqDist(text)
    hi_freq = list(filter(lambda x: x[1]>10000,fdist.items()))  
    text = [w for w in word_list if not w in stop_words and not w in hi_freq]  #high frequency (more than 5000 occurrencies) words are removed
    text = ' '.join(text)
    return text
    
def datasetcleansing(dataset):
    input_file = dataset
    output_file = 'clean'+dataset
    # Open the input and temporary TSV files
    with open(input_file, 'r', newline='', encoding='utf-8') as input_tsv, open(output_file, 'w', newline='', encoding='utf-8') as outfile:
        tsv_reader = csv.reader(input_tsv, delimiter=';')
        tsv_writer = csv.writer(outfile, delimiter=';')

        for row in tsv_reader:
            if '0' not in row and '0.0' not in row:
                tsv_writer.writerow(row)
    print(f"Filtered data written to {output_file}")    
    
#This function extracts the in- and outgoing Wikipedia links from the DBpedia main page

def expand_entity(entity):

    endpoint = "http://dbpedia.org/sparql"
    input_entity_uri = "http://dbpedia.org/resource/"+str(entity)
    sparql_query = f"""
        PREFIX dbo: <http://dbpedia.org/ontology/>
       
        SELECT DISTINCT ?relatedEntity
        WHERE {{
        {{
          <{input_entity_uri}> dbo:wikiPageWikiLink ?relatedEntity.
        }}
        UNION
        {{
          ?relatedEntity dbo:wikiPageWikiLink <{input_entity_uri}>.
        }}
        }}
    """

    # Set up the SPARQLWrapper
    sparql = SPARQLWrapper(endpoint)
    sparql.setQuery(sparql_query)
    sparql.setReturnFormat(JSON)

    # Execute the SPARQL query
    results = sparql.query().convert()

    # Extract the related entities from the results
    related_entities = [result['relatedEntity']['value'] for result in results['results']['bindings']]

    return related_entities


# entity = "Aristotle"
# related_entities = expand_entity(entity)

# print(f"Entities related to {entity}:")
# for entity in related_entities:
#     print(entity)

def sort_tsv(input_file):
    output_file = 'sorted_output.csv'

    df = pd.read_csv(input_file, sep=';',error_bad_lines=False, warn_bad_lines=True)
    df.sort_values(by=df.columns[1], inplace=True)

    df.to_csv(output_file, index=False)

#Usage example

# input_file = 'C:\\Users\\Palma\\Desktop\\PHD\\ISAAKbackup\\ISAAKx\\clkstrdataset8.tsv'
# sort_tsv(input_file)