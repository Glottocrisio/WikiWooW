# Copyright (c) 2025 Palma. All rights reserved.
# Author: Palma.
#
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
import time
import Interestingness as I
import sys
import similarity_patch as sp

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
    original_data.to_csv("data/datasets/updated_data.csv", index=False)

def file_partitioning(file_path, num_partitions):

    df = pd.read_csv(file_path, sep='\t')
    rows_per_partition = len(df) // num_partitions
    partitions = [df.iloc[i:i+rows_per_partition] for i in range(0, len(df), rows_per_partition)]

    for i, partition in enumerate(partitions):
        output_file_path = f'data/partitions/output_partition_alex{i+1}.tsv'
        partition.to_csv(output_file_path, sep='\t', index=False)

    print("Partitioning complete.")
    
#file_partitioning('C:\\Users\\Palma\\Desktop\\PHD\\ISAAKbackup\\ISAAKx\\clickstream-hrwiki-2025-03.tsv', 5)

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


def filter_wikilinks(input_filename='merged_knowledge_graph_triples.csv'):

    input_path = os.path.join('/content', input_filename)

    print(f"Reading file from: {input_path}")
    df = pd.read_csv(input_path)

    # Filter triples with the specified predicate
    wikilink_predicate = "http://dbpedia.org/ontology/wikiPageWikiLink"
    filtered_df = df[df['predicate'] == wikilink_predicate]

    # Save filtered data to new CSV
    output_file = 'wikilinks.csv'
    output_path = os.path.join('/content', output_file)
    filtered_df.to_csv(output_path, index=False)

    # Print statistics
    print(f"\nStatistics:")
    print(f"Total triples in input: {len(df)}")
    print(f"Triples with wikiPageWikiLink: {len(filtered_df)}")
    print(f"Filtered data saved to: {output_path}")

# Run the filter function
#try:
#    filter_wikilinks()
#    print("\nProcessing completed successfully!")
#except Exception as e:
#    print(f"An error occurred: {str(e)}")

#import pandas as pd
#import time

def rebuild_threads(input_file='output/reports/experiment3.txt', clickstream_file='data/datasets/clickstream_matches.csv', output_file='data/datasets/clickstream_threads.csv'):
    """
    Reconstruct three-entity threads (A→B→C) from the original pair file
    and enrich them with clickstream values.
    
    The input file contains pairs derived from threads of three entities:
    For each thread A→B→C, there are two consecutive rows: A;B and B;C.
    """
    start_time = time.time()
    print("=== Rebuilding Three-Entity Threads ===\n")

    # --- 1. Read original pairs to recover thread structure ---
    print(f"Reading original pairs from: {input_file}")
    if input_file.endswith('.csv'):
        pairs_df = pd.read_csv(input_file)
    elif input_file.endswith('.txt'):
        pairs_df = pd.read_csv(input_file, sep=';', usecols=[0, 1],
                               names=['subject', 'object'], header=0)
    else:
        raise ValueError(f"Unsupported format: {input_file}")

    # Clean entity names (normalize spaces to underscores, strip URIs)
    for col in ['subject', 'object']:
        pairs_df[col] = (pairs_df[col]
                         .str.strip()
                         .str.replace('http://dbpedia.org/resource/', '', regex=False)
                         .str.replace(' ', '_'))

    print(f"Total pairs loaded: {len(pairs_df)}")

    # --- Identify seed entities (most frequent = thread endpoints) ---
    from collections import Counter
    all_entities = list(pairs_df['subject']) + list(pairs_df['object'])
    entity_freq = Counter(all_entities)

    print(f"\nTop 10 most frequent entities:")
    for ent, freq in entity_freq.most_common(10):
        print(f"  {ent}: {freq}")

    # --- Reconstruct threads using frequency logic ---
    # For each pair, each entity is either a "seed" (high freq, = endpoint)
    # or a "middle" (low freq, = bridge). Threads are: seed → middle → seed.
    # We group pairs by their shared middle entity.

    # Build adjacency: for each entity, collect its partners
    from collections import defaultdict
    adjacency = defaultdict(set)
    for _, row in pairs_df.iterrows():
        adjacency[row['subject']].add(row['object'])
        adjacency[row['object']].add(row['subject'])

    # A middle entity connects exactly two seeds.
    # Identify seeds: entities with frequency above median
    freq_values = list(entity_freq.values())
    freq_threshold = sorted(freq_values, reverse=True)[min(len(freq_values) - 1,
                            max(1, len(freq_values) // 4))]  # top 25% as seeds
    seed_entities = {ent for ent, freq in entity_freq.items() if freq >= freq_threshold}

    print(f"\nFrequency threshold for seeds: {freq_threshold}")
    print(f"Seed entities identified: {len(seed_entities)}")

    # Build threads: for each non-seed entity, find its seed neighbors
    threads = []
    seen_threads = set()
    for middle_ent in adjacency:
        if middle_ent in seed_entities:
            continue
        neighbors = adjacency[middle_ent]
        seed_neighbors = [n for n in neighbors if n in seed_entities]
        # Create a thread for each pair of seed neighbors
        for i, seed1 in enumerate(seed_neighbors):
            for seed2 in seed_neighbors[i + 1:]:
                thread_key = tuple(sorted([seed1, seed2]) + [middle_ent])
                if thread_key not in seen_threads:
                    seen_threads.add(thread_key)
                    threads.append({
                        'entity1': seed1,
                        'entity2': middle_ent,
                        'entity3': seed2,
                    })

    threads_df = pd.DataFrame(threads)
    print(f"Threads reconstructed: {len(threads_df)}")

    # --- 2. Read clickstream matches and build lookup ---
    print(f"\nReading clickstream matches from: {clickstream_file}")
    cs_df = pd.read_csv(clickstream_file)

    # Strip URI prefix for matching
    for col in ['entity1', 'entity2']:
        cs_df[col] = cs_df[col].str.replace('http://dbpedia.org/resource/', '', regex=False)

    # Build lookup dict: (entity1, entity2) -> clickstream value
    cs_lookup = dict(zip(zip(cs_df['entity1'], cs_df['entity2']),
                         cs_df['ClickstreamE1-E2']))
    print(f"Clickstream pairs loaded: {len(cs_lookup)}")

    # --- 3. Merge clickstream values into threads ---
    # Check both directions since clickstream is directional
    threads_df['ClickstreamE1-E2'] = threads_df.apply(
        lambda r: cs_lookup.get((r['entity1'], r['entity2']),
                  cs_lookup.get((r['entity2'], r['entity1']), 0)), axis=1)
    threads_df['ClickstreamE2-E3'] = threads_df.apply(
        lambda r: cs_lookup.get((r['entity2'], r['entity3']),
                  cs_lookup.get((r['entity3'], r['entity2']), 0)), axis=1)

    # Add URI prefix back
    for col in ['entity1', 'entity2', 'entity3']:
        threads_df[col] = 'http://dbpedia.org/resource/' + threads_df[col]

    # Reorder columns to match requested format
    threads_df = threads_df[['entity1', 'entity2', 'ClickstreamE1-E2',
                             'entity2', 'entity3', 'ClickstreamE2-E3']]

    # Rename duplicate 'entity2' column for valid CSV output
    threads_df.columns = ['entity1', 'entity2', 'ClickstreamE1-E2',
                          'entity2_', 'entity3', 'ClickstreamE2-E3']

    # --- 4. Save ---
    print(f"\nSaving to: {output_file}")
    # Write with the desired headers (entity2 appears twice)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('entity1,entity2,ClickstreamE1-E2,entity2,entity3,ClickstreamE2-E3\n')
        for _, row in threads_df.iterrows():
            f.write(f"{row['entity1']},{row['entity2']},{row['ClickstreamE1-E2']},"
                    f"{row['entity2_']},{row['entity3']},{row['ClickstreamE2-E3']}\n")

    elapsed = time.time() - start_time
    print(f"\n=== Done in {elapsed:.2f}s ===")
    print(f"Threads with at least one clickstream match: "
          f"{len(threads_df[(threads_df['ClickstreamE1-E2'] != 0) | (threads_df['ClickstreamE2-E3'] != 0)])}")
    print(f"Total threads saved: {len(threads_df)}")
    print(f"\nSample output:")
    print(threads_df.head(5).to_string(index=False))


#if __name__ == "__main__":
 #   rebuild_threads()

"""
recalculate_nulls.py
====================
Reads the output CSV, identifies rows where DBpediaSimilarityEnt1Ent2 or
PalmaInterestingness6Ent1Ent2 (or however they are named) contain nulls/zeros,
recalculates only those cells, and overwrites the file.
"""




def recalculate_nulls(filepath):
    start_time = time.time()
    print(f"=== Recalculating null columns in: {filepath} ===\n")

    # Read the file
    df = pd.read_csv(filepath, sep=';')
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}\n")

    # --- Auto-detect the null columns ---
    # Find columns that contain nulls, NaN, 0, or empty strings
    null_cols = []
    for col in df.columns:
        null_count = df[col].isna().sum() + (df[col] == 0).sum() + (df[col] == '').sum()
        if null_count > 0:
            print(f"  Column '{col}': {null_count} null/zero values out of {len(df)}")
            null_cols.append(col)

    if not null_cols:
        print("No null columns found. Nothing to recalculate.")
        return

    print(f"\nColumns to recalculate: {null_cols}")

    # --- Identify entity columns ---
    # Try common naming patterns
    ent1_col = None
    ent2_col = None
    for col in df.columns:
        col_lower = col.lower().strip()
        if col_lower in ['Entity1', 'ent1', 'entity_1', 'e1']:
            ent1_col = col
        elif col_lower in ['Entity2', 'ent2', 'entity_2', 'e2']:
            ent2_col = col

    if not ent1_col or not ent2_col:
        # Fallback: assume first two columns are entities
        ent1_col = df.columns[0]
        ent2_col = df.columns[1]

    print(f"\nUsing entity columns: '{ent1_col}' and '{ent2_col}'")

    # --- Recalculate ---
    recalc_count = 0
    total_rows = len(df)

    for idx, row in df.iterrows():
        ent1 = str(row[ent1_col])
        ent2 = str(row[ent2_col])

        row_changed = False

        for col in null_cols:
            val = row[col]
            is_null = pd.isna(val) or val == 0 or val == '' or val == 0.0

            if not is_null:
                continue

            col_lower = col.lower().replace(' ', '').replace('_', '')
            new_val = None

            # Match column name to the appropriate function
            if 'dbpediasimilarity' in col_lower and 'relatedness' not in col_lower:
                print(f"  [{idx+1}/{total_rows}] Recalculating DBpediaSimilarity for {ent1} / {ent2}...")
                new_val = sp.fDBpediaSimilarity(ent1, ent2)

            elif 'dbpediarelatedness' in col_lower:
                print(f"  [{idx+1}/{total_rows}] Recalculating DBpediaRelatedness for {ent1} / {ent2}...")
                new_val = I.fDBpediaRelatedness(ent1, ent2)

            elif 'cosinesimilarity' in col_lower or 'cosinsimilarity' in col_lower:
                print(f"  [{idx+1}/{total_rows}] Recalculating CosineSimilarity for {ent1} / {ent2}...")
                new_val = I.CosineSimilarity("en", ent1, ent2)

            elif 'interestingness6' in col_lower or 'lukasiewicz' in col_lower:
                print(f"  [{idx+1}/{total_rows}] Recalculating Interestingness6 (Lukasiewicz) for {ent1} / {ent2}...")
                new_val = I.palma_interestingness_lukasiewicz(ent1, ent2)

            elif 'interestingness1' in col_lower and '10' not in col_lower:
                new_val = I.palma_interestingness(ent1, ent2)

            elif 'interestingness2' in col_lower:
                new_val = I.palma_interestingness2(ent1, ent2)

            elif 'interestingness3' in col_lower:
                new_val = I.palma_interestingness3(ent1, ent2)

            elif 'interestingness4' in col_lower:
                new_val = I.palma_interestingness4(ent1, ent2)

            elif 'interestingness5' in col_lower:
                new_val = I.palma_interestingness5(ent1, ent2)

            elif 'interestingnessweighted' in col_lower or 'interestingness7' in col_lower:
                new_val = I.palma_interestingness_weighted(ent1, ent2)

            if new_val is not None:
                df.at[idx, col] = new_val
                row_changed = True

        if row_changed:
            recalc_count += 1

    # --- Overwrite ---
    print(f"\n=== Saving: {recalc_count} rows updated ===")
    df.to_csv(filepath, sep=';', index=False, encoding='utf-8')

    elapsed = time.time() - start_time
    print(f"Done in {elapsed:.2f}s. File overwritten: {filepath}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = 'data/datasets/exp3_addin.csv_dataset_corrected_final.csv'

    recalculate_nulls(filepath)