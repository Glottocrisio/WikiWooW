# Copyright (c) 2025 Palma. All rights reserved.
# Author: Palma.
#
# This script retrieves the entity couples harvested through "ExtractWikiPageWikiLink" in the Wikimedia
# Clickstream data dump.

import pandas as pd
import time

def preprocess_wikilinks(filepath='experiment3.txt'):
    """Load and preprocess wikilinks data (supports .csv and .txt semicolon-separated)"""
    print("\n=== Starting Wikilinks Preprocessing ===")
    start_time = time.time()

    print(f"Reading document: {filepath}")
    if filepath.endswith('.csv'):
        wiki_df = pd.read_csv(filepath)
    elif filepath.endswith('.txt'):
        # Only read the first two columns (subject and object),
        # ignoring any trailing semicolons / extra empty columns
        wiki_df = pd.read_csv(filepath, sep=';', usecols=[0, 1],
                              names=['subject', 'object'], header=0)
    else:
        raise ValueError(f"Unsupported file format: {filepath}. Use .csv or .txt")

    print(f"Initial wikilinks shape: {wiki_df.shape}")

    # Drop any rows where subject or object is missing
    wiki_df.dropna(subset=['subject', 'object'], inplace=True)

    print("Cleaning entity names...")
    # Strip DBpedia URI prefix if present, and normalize spaces to underscores
    for col in ['subject', 'object']:
        wiki_df[col] = (wiki_df[col]
                        .str.strip()
                        .str.replace('http://dbpedia.org/resource/', '', regex=False)
                        .str.replace(' ', '_'))

    print("Creating lookup set...")
    lookup_set = set(zip(wiki_df['subject'], wiki_df['object']))

    end_time = time.time()
    print(f"Wikilinks preprocessing completed in {end_time - start_time:.2f} seconds")
    print(f"Number of unique entity pairs: {len(lookup_set)}")
    print(f"Sample pairs: {list(lookup_set)[:5]}")
    print("=== Preprocessing Complete ===\n")

    return lookup_set


def process_clickstream(lookup_set, filepath='clickstream-enwiki-2023-11.tsv'):
    """Process clickstream using vectorized set lookup (supports .tsv and .tsv.gz)"""
    all_matches = []
    chunk_size = 1_000_000
    total_rows_processed = 0
    start_time = time.time()

    print("\n=== Starting Clickstream Processing ===")
    print(f"Processing: {filepath}")
    print(f"Chunk size: {chunk_size:,} rows")

    read_kwargs = {
        'sep': '\t',
        'names': ['prev_id', 'curr_id', 'type', 'n'],
        'on_bad_lines': 'skip',
        'chunksize': chunk_size,
        'dtype': {'prev_id': str, 'curr_id': str, 'type': str, 'n': str},
    }
    if filepath.endswith('.gz'):
        read_kwargs['compression'] = 'gzip'

    try:
        chunks = pd.read_csv(filepath, **read_kwargs)

        for chunk_num, chunk in enumerate(chunks, 1):
            chunk_start_time = time.time()
            initial_chunk_size = len(chunk)

            # Drop rows with missing prev_id or curr_id
            chunk = chunk.dropna(subset=['prev_id', 'curr_id'])

            # Vectorized lookup: build tuples from the chunk,
            # then check membership against the lookup set
            pairs = list(zip(chunk['prev_id'], chunk['curr_id']))
            mask = [p in lookup_set for p in pairs]
            matching_rows = chunk.loc[mask]

            if len(matching_rows) > 0:
                matches_chunk = pd.DataFrame({
                    'entity1': 'http://dbpedia.org/resource/' + matching_rows['prev_id'],
                    'entity2': 'http://dbpedia.org/resource/' + matching_rows['curr_id'],
                    'ClickstreamE1-E2': matching_rows['n'].values,
                })
                all_matches.append(matches_chunk)

            total_rows_processed += initial_chunk_size
            chunk_time = time.time() - chunk_start_time

            print(f"Chunk {chunk_num}: "
                  f"{initial_chunk_size:,} rows | "
                  f"{len(matching_rows):,} matches | "
                  f"{chunk_time:.2f}s | "
                  f"{initial_chunk_size / max(chunk_time, 0.001):,.0f} rows/s")

    except Exception as e:
        print(f"\nError processing chunk: {str(e)}")
        raise

    end_time = time.time()
    total_time = end_time - start_time

    # Concatenate all match DataFrames at once
    if all_matches:
        matches_df = pd.concat(all_matches, ignore_index=True)
    else:
        matches_df = pd.DataFrame(columns=['entity1', 'entity2', 'ClickstreamE1-E2'])

    print(f"\n=== Clickstream Processing Complete ===")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Total rows processed: {total_rows_processed:,}")
    print(f"Total matches found: {len(matches_df):,}")
    print(f"Average speed: {total_rows_processed / max(total_time, 0.001):,.0f} rows/second")

    return matches_df


def main():
    overall_start_time = time.time()
    print("=== Starting Knowledge Graph Matching Process ===")

    try:
        # --- Configure file paths here ---
        wikilinks_file = 'output/reports/experiment3.txt'
        clickstream_file = 'data/datasets/clickstream-enwiki-2023-11.tsv'
        output_file = 'data/datasets/clickstream_matches.csv'
        # ----------------------------------

        lookup_set = preprocess_wikilinks(wikilinks_file)
        matches_df = process_clickstream(lookup_set, clickstream_file)

        if len(matches_df) > 0:
            print("\n=== Saving Results ===")
            matches_df.to_csv(output_file, index=False)
            print(f"Total matches found: {len(matches_df):,}")
            print(f"Results saved to: {output_file}")
            print(f"\nSample results:")
            print(matches_df.head(10).to_string(index=False))
        else:
            print("\nNo matches found!")

    except Exception as e:
        print(f"\nCritical error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

    finally:
        total_runtime = time.time() - overall_start_time
        print(f"\nTotal runtime: {total_runtime:.2f} seconds ({total_runtime / 60:.2f} minutes)")


if __name__ == "__main__":
    main()