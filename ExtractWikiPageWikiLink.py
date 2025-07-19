# This function takes in input a CSV file containing an heterogeneous knowledge graph
#  (triples contained in 'subject', 'predicate', 'object') and extracts only the entity-couples 
# connected by the property 'wikiPageWikiLink' (i.e., the links between Wikipedia pages).

import pandas as pd
import os

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
try:
    filter_wikilinks()
    print("\nProcessing completed successfully!")
except Exception as e:
    print(f"An error occurred: {str(e)}")
