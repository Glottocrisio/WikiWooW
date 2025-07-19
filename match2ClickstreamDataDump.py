# This script retrieves the entity couples harvested through "ExtractWikiPageWikiLink" in the Wikimedia
# Clickstream data dump.


import pandas as pd
import gzip
from tqdm import tqdm
import numpy as np
import time

def preprocess_wikilinks():
    """Load and preprocess wikilinks data"""
    print("\n=== Starting Wikilinks Preprocessing ===")
    start_time = time.time()

    print("Reading wikilinks.csv...")
    wiki_df = pd.read_csv('/content/wikilinks.csv')
    print(f"Initial wikilinks shape: {wiki_df.shape}")

    print("Cleaning entity names...")
    wiki_df['subject_clean'] = wiki_df['subject'].str.replace('http://dbpedia.org/resource/', '')
    wiki_df['object_clean'] = wiki_df['object'].str.replace('http://dbpedia.org/resource/', '')

    print("Creating lookup dictionary...")
    lookup_dict = set(zip(wiki_df['subject_clean'], wiki_df['object_clean']))

    end_time = time.time()
    print(f"Wikilinks preprocessing completed in {end_time - start_time:.2f} seconds")
    print(f"Number of unique entity pairs: {len(lookup_dict)}")
    print("=== Preprocessing Complete ===\n")

    return lookup_dict

def process_clickstream_in_memory(lookup_dict):
    """Process clickstream using binary search approach"""
    matches = []
    chunk_size = 1000000
    total_rows_processed = 0
    start_time = time.time()

    print("\n=== Starting Clickstream Processing ===")
    print(f"Processing /content/clickstream-enwiki-2024-10.tsv.gz")
    print(f"Chunk size: {chunk_size:,} rows")

    try:
        # Read with minimal assumptions about the file structure
        chunks = pd.read_csv('/content/clickstream-enwiki-2024-10.tsv.gz',
                            compression='gzip',
                            sep='\t',
                            names=['prev_id', 'curr_id', 'type', 'n'],
                            on_bad_lines='skip',
                            chunksize=chunk_size)

        for chunk_num, chunk in enumerate(chunks, 1):
            chunk_start_time = time.time()
            initial_chunk_size = len(chunk)

            # Filter valid rows
            chunk = chunk[chunk['prev_id'].str.contains('^[A-Za-z0-9_]+$', na=False) &
                         chunk['curr_id'].str.contains('^[A-Za-z0-9_]+$', na=False)]

            # Check matches
            mask = chunk.apply(lambda x: (x['prev_id'], x['curr_id']) in lookup_dict, axis=1)
            matching_rows = chunk[mask]

            # Process matching rows
            for _, row in matching_rows.iterrows():
                matches.append({
                    'entity1': f"http://dbpedia.org/resource/{row['prev_id']}",
                    'entity2': f"http://dbpedia.org/resource/{row['curr_id']}",
                    'ClickstreamE1-E2': row['n']
                })

            # Update progress
            total_rows_processed += initial_chunk_size
            chunk_time = time.time() - chunk_start_time

            print(f"\nChunk {chunk_num} Statistics:")
            print(f"- Rows processed: {initial_chunk_size:,}")
            print(f"- Valid rows: {len(chunk):,}")
            print(f"- Matches found in chunk: {len(matching_rows):,}")
            print(f"- Total matches so far: {len(matches):,}")
            print(f"- Total rows processed: {total_rows_processed:,}")
            print(f"- Chunk processing time: {chunk_time:.2f} seconds")
            print(f"- Average processing speed: {initial_chunk_size/chunk_time:,.0f} rows/second")

    except Exception as e:
        print(f"\nError processing chunk: {str(e)}")
        raise

    end_time = time.time()
    total_time = end_time - start_time

    print("\n=== Clickstream Processing Complete ===")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Total rows processed: {total_rows_processed:,}")
    print(f"Total matches found: {len(matches):,}")
    print(f"Average processing speed: {total_rows_processed/total_time:,.0f} rows/second")

    return matches

def main():
    overall_start_time = time.time()
    print("=== Starting Knowledge Graph Matching Process ===")

    try:
        # Preprocess wikilinks
        lookup_dict = preprocess_wikilinks()

        # Process clickstream data
        matches = process_clickstream_in_memory(lookup_dict)

        # Save results
        if matches:
            print("\n=== Saving Results ===")
            print("Creating DataFrame...")
            matches_df = pd.DataFrame(matches)

            print("Writing to CSV...")
            matches_df.to_csv('/content/clickstream_matches.csv', index=False)

            print("\n=== Process Complete ===")
            print(f"Total matches found: {len(matches):,}")
            print("Results saved to: /content/clickstream_matches.csv")
        else:
            print("\nNo matches found!")

    except Exception as e:
        print(f"\nCritical error occurred: {str(e)}")

    finally:
        overall_end_time = time.time()
        total_runtime = overall_end_time - overall_start_time
        print(f"\nTotal runtime: {total_runtime:.2f} seconds ({total_runtime/60:.2f} minutes)")

if __name__ == "__main__":
    main()
