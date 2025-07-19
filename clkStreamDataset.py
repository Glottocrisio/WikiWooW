#This module implements the creation of a Dataset derived from wikipedia clickstream-data
#data dump including all other measures (centrality, popularity, similarity)

import requests
import matplotlib.pyplot as plt
import metrics as me
import math
import pandas as pd
import csv
import Palma_Interestingness as pal
import math
import os
import tools as t



def enhance_entity_pairs_with_metrics(input_csv, output_csv):
    """
    Process a CSV file containing entity pairs and enhance it with interestingness metrics.
    
    Args:
        input_csv: Path to the input CSV file with entity pairs
        output_csv: Path to save the enhanced output CSV
    """
    # Read the input CSV
    print(f"Reading input file: {input_csv}")
    
    # Try different encodings if needed
    try:
        df = pd.read_csv(input_csv, encoding='utf-8', on_bad_lines='skip', delimiter=';')
    except UnicodeDecodeError:
        df = pd.read_csv(input_csv, encoding='latin1')
    
    # Identify the entity columns
    entity_columns = [col for col in df.columns if 'entity' in col.lower()]
    
    # Create output file with headers
    with open(output_csv, 'w', newline='', encoding='utf-8') as output_file:
        fieldnames = ['Entity1', 'Entity2', 'ClickstreamEnt1Ent2', 
                     'PopularityEnt1', 'PopularityEnt2',  # Keep original popularity columns
                     'PageRankEnt1', 'PageRankEnt2',      # New PageRank columns
                     'PageViewEnt1', 'PageViewEnt2',      # PageView columns
                     'PopularityDiff', 'PopularitySum', 'CosineSimilarityEnt1Ent2', 
                     'DBpediaSimilarityEnt1Ent2', 'DBpediaRelatednessEnt1Ent2', 
                     'PalmaInterestingnessEnt1Ent2', 'PalmaInterestingness2Ent1Ent2',
                     'PalmaInterestingness3Ent1Ent2', 'PalmaInterestingness4Ent1Ent2',
                     'PalmaInterestingness5Ent1Ent2']  # Additional PalmaInterestingness columns
        
        # Add original columns that aren't already in fieldnames
        for col in df.columns:
            if col not in entity_columns and col not in fieldnames:
                fieldnames.append(col)
        
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        
        # Process each row
        total_rows = len(df)
        for idx, row in df.iterrows():
            if idx % 10 == 0:
                print(f"Processing row {idx+1}/{total_rows}")
            
            ent1 = row[entity_columns[0]]
            ent2 = row[entity_columns[1]]
            
            # Skip rows with missing entities
            if pd.isna(ent1) or pd.isna(ent2):
                print(f"Skipping row {idx+1} due to missing entity values")
                continue
            
            # Convert entities to string and clean them
            ent1 = str(ent1).replace(' ', '_')
            ent2 = str(ent2).replace(' ', '_')
            
            # Initialize output row with entity values
            output_row = {
                'Entity1': ent1,
                'Entity2': ent2
            }
            
            # Copy original data to output
            for col in df.columns:
                if col not in entity_columns and col in fieldnames:
                    output_row[col] = row[col]
            
            
            pagerank1 = pal.WikifierPageRank(ent1)
                
            pagerank2 = pal.WikifierPageRank(ent2)
                
            output_row['PageRankEnt1'] = pagerank1
            output_row['PageRankEnt2'] = pagerank2

            # Calculate PageView metrics
            try:
                pageview1 = pal.singleclickstream(ent1)
                pageview2 = pal.singleclickstream(ent2)
                
                output_row['PageViewEnt1'] = pageview1
                output_row['PageViewEnt2'] = pageview2
            except Exception as e:
                print(f"Error calculating PageViews for {ent1}, {ent2}: {e}")
                output_row['PageViewEnt1'] = 0
                output_row['PageViewEnt2'] = 0
            
            # Calculate Popularity metrics using PageView/PageRank formula
            try:
                # Calculate popularity as PageView / PageRank
                if float(output_row['PageRankEnt1']) > 0:
                    popularity1 = output_row['PageViewEnt1'] / output_row['PageRankEnt1']
                else:
                    popularity1 = 0
                
                if float(output_row['PageRankEnt2']) > 0:
                    popularity2 = output_row['PageViewEnt2'] / output_row['PageRankEnt2']
                else:
                    popularity2 = 0
                
                output_row['PopularityEnt1'] = round(popularity1, 2)
                output_row['PopularityEnt2'] = round(popularity2, 2)
                output_row['PopularityDiff'] = round(abs(popularity1 - popularity2), 2)
                output_row['PopularitySum'] = round(popularity1 + popularity2, 2)
            except Exception as e:
                print(f"Error calculating Popularity for {ent1}, {ent2}: {e}")
                output_row['PopularityEnt1'] = 0
                output_row['PopularityEnt2'] = 0
                output_row['PopularityDiff'] = 0
                output_row['PopularitySum'] = 0
            
            
            # Calculate similarity metrics
            try:
                output_row['CosineSimilarityEnt1Ent2'] = pal.CosineSimilarity("en", ent1, ent2)
            except Exception as e:
                print(f"Error calculating cosine similarity for {ent1}, {ent2}: {e}")
                output_row['CosineSimilarityEnt1Ent2'] = 0.01
                
            try:
                output_row['DBpediaSimilarityEnt1Ent2'] = pal.fDBpediaSimilarity(ent1, ent2)
            except Exception as e:
                print(f"Error calculating DBpedia similarity for {ent1}, {ent2}: {e}")
                output_row['DBpediaSimilarityEnt1Ent2'] = 0.01
                
            try:
                output_row['DBpediaRelatednessEnt1Ent2'] = pal.fDBpediaRelatedness(ent1, ent2)
            except Exception as e:
                print(f"Error calculating DBpedia relatedness for {ent1}, {ent2}: {e}")
                output_row['DBpediaRelatednessEnt1Ent2'] = 0.01
            
            # Calculate Palma interestingness (original)
            try:
                # Use the palma_interestingness formula directly to handle exceptions better
                if (output_row['PopularitySum'] > 0 and 
                    output_row['CosineSimilarityEnt1Ent2'] > 0):
                    
                    pop = math.log(output_row['PopularitySum'] + output_row['PopularityDiff']) + 1
                    csim = math.log((output_row['CosineSimilarityEnt1Ent2'] + output_row['DBpediaSimilarityEnt1Ent2'])/2)
                    ksim = math.log(output_row['DBpediaRelatednessEnt1Ent2'] + 0.1)
                    
                    palmint = (pop * abs(csim - ksim))
                    # Normalize by clickstream if available
                    if output_row['ClickstreamEnt1Ent2'] > 1:
                        palmint = round(palmint/math.log10(output_row['ClickstreamEnt1Ent2']), 2)
                    
                    output_row['PalmaInterestingnessEnt1Ent2'] = palmint
                else:
                    output_row['PalmaInterestingnessEnt1Ent2'] = 0
            except Exception as e:
                print(f"Error calculating Palma interestingness for {ent1}, {ent2}: {e}")
                output_row['PalmaInterestingnessEnt1Ent2'] = 0
            
            # Calculate additional Palma interestingness metrics (2, 3, 4, 5)
            try:
                output_row['PalmaInterestingness2Ent1Ent2'] = pal.palma_interestingness2(ent1, ent2)
            except Exception as e:
                print(f"Error calculating Palma interestingness 2 for {ent1}, {ent2}: {e}")
                output_row['PalmaInterestingness2Ent1Ent2'] = 0
                 
            try:
                output_row['PalmaInterestingness3Ent1Ent2'] = pal.palma_interestingness3(ent1, ent2)
            except Exception as e:
                print(f"Error calculating Palma interestingness 3 for {ent1}, {ent2}: {e}")
                output_row['PalmaInterestingness3Ent1Ent2'] = 0
                
            try:
                output_row['PalmaInterestingness4Ent1Ent2'] = pal.palma_interestingness4(ent1, ent2)
            except Exception as e:
                print(f"Error calculating Palma interestingness 4 for {ent1}, {ent2}: {e}")
                output_row['PalmaInterestingness4Ent1Ent2'] = 0
                
            try:
                output_row['PalmaInterestingness5Ent1Ent2'] = pal.palma_interestingness5(ent1, ent2)
            except Exception as e:
                print(f"Error calculating Palma interestingness 5 for {ent1}, {ent2}: {e}")
                output_row['PalmaInterestingness5Ent1Ent2'] = 0
            
            # Write row to output file
            writer.writerow(output_row)
    
    print(f"Processing completed. Output saved to {output_csv}")
    return output_csv

##List of adopted heuristics
#pop(e) = clickstream(e)/pagerankcentrality(e)
#palmainterestingness(e1-e2) = (ln(popsum) - abs(popdif)) *
# ln (|cosinsimilaritylabelse1labelsee2 − yagosimilaritye1e2|)
# (to be extended with the relation. adapted from text2story2024)

## for the downstream task of the levensthein distance calculation one can use a
## pretrained LLM (like glove).

## Preprocessing

def preprocessclkstrdataset(clkstrdump, clkstrdataset):
    input_file = "C:\\Users\\Palma\\Desktop\\PHD\\ISAAKx\\"+str(clkstrdump)
    output_file = "C:\\Users\\Palma\\Desktop\\PHD\\ISAAKx\\"+str(clkstrdataset)
    clkstrd = str(clkstrdataset)
    # If dataset-file does not exist already, it creates it
    if os.path.exists(input_file) and not os.path.exists(clkstrdataset):
        with open(input_file, 'r', newline='', encoding='utf-8') as infile:
            reader = csv.reader(infile, delimiter='\t')
            # Remove all lines containing the word "other"
            columns = [[row[0], row[1], row[3]] for row in reader if 'other' not in row[0]]
        with open(clkstrdataset, "w", newline='', encoding='utf-8') as clkstrdataset:
            writer = csv.writer(clkstrdataset, delimiter='\t')
            for value in columns:
                writer.writerow([value])
            clkstrdataset.close()
        # Remove "[]" 
        # Replace "" with '
        with open(output_file, "r+", newline='', encoding='utf-8') as output_file:
            clkstrstring = output_file.read()
            clkstrstring = clkstrstring.replace("[","").replace("]","").replace("\"\"","'").replace(",",";").replace("\"","").replace(" ","")
            output_file.close()  
        with open("C:\\Users\\Palma\\Desktop\\PHD\\ISAAKx\\"+str(clkstrd), "w", newline='', encoding='utf-8') as output:
            output.write(clkstrstring)
            output.close()
 

#This function queries the wikimedia clickstream data-dump and finds the couplets as in the list

def findwikiPageWikiLink(input_file, output_file, entity):
        # Open the input and temporary TSV files
    with open(input_file, 'r', newline='', encoding='utf-8') as input_tsv, open(output_file, 'w', newline='', encoding='utf-8') as output_tsv:
        # Create TSV reader and writer
        tsv_reader = csv.reader(input_tsv, delimiter=';')
        tsv_writer = csv.writer(output_tsv, delimiter=';')
        
        # Write header to the temporary TSV file
        header = ['Entity1', 'Entity2', 'ClickstreamEnt1Ent2', 'PopularityEnt1','PopularityEnt2', 
                  'PopularityDiff', 'PopularitySum', 'CosineSimilarityEnt1Ent2', 'DBpediaSimilarityEnt1Ent2', 'DBpediaRelatednessEnt1Ent2', 'PalmaInterestingnessEnt1Ent2']
        tsv_writer.writerow(header)

        next(tsv_reader)
        # Process each row in the dataset
        for row in tsv_reader:
            # with this loop we bring back the clickstream column as an int
            #row[2] = int(row[2])
            
            ent1, ent2 = row[:2]
            if ent1  == entity or ent2 == entity:
                try:
                    tsv_writer.writerow(row)
                except Exception as e:
                    continue
            else:
                continue


def addPopularityclkstrdataset(clkstrdataset):
    
    input_file = clkstrdataset
    temp_file = 'pop'+clkstrdataset

    # Open the input and temporary TSV files
    with open(input_file, 'r', newline='', encoding='utf-8') as input_tsv, open(temp_file, 'w', newline='') as temp_tsv:
        # Create TSV reader and writer
        tsv_reader = csv.reader(input_tsv, delimiter=';')
        tsv_writer = csv.writer(temp_tsv, delimiter=';')

        # Write header to the temporary TSV file
        header = ['Entity1', 'Entity2', 'ClickstreamEnt1Ent2', 'PopularityEnt1','PopularityEnt2', 
                  'PopularityDiff', 'PopularitySum', 'CorpusSimilarityEnt1Ent2', 'CosineSimilarityEnt1Ent2', 'WupSimilarityEnt1Ent2', 'PathSimilarityEnt1Ent2', 
                 'DBpediaSimilarityEnt1Ent2', 'DBpediaRelatednessEnt1Ent2', 'PalmaInterestingnessEnt1Ent2']
        tsv_writer.writerow(header)
        # Process each row in the dataset

        for row in tsv_reader:
            # with this loop we bring back the clickstream column as an int
            #row[2] = int(row[2])
            try:
                row[2] = int(row[2].replace("'",""))
            except Exception as e:
                continue
            ent1, ent2 = row[:2]
            
            pope1 = pal.entpopularity(ent1)
            
            pope2 = pal.entpopularity(ent2)
            popdiff = round(abs(pope1 - pope2),2)
            popsum = round(abs(pope1 + pope2),2)
            
             # with this line the temp_file is prevented to contain spurious
            if pope1==0 or pope2 == 0:
                pass
            else:
                
                row += [pope1, pope2, popdiff, popsum]
                try:
                    tsv_writer.writerow(row)
                except Exception as e:
                    continue
            
            # Hereby we add to the next two columns popdiff and popsum

    print(f"Processing completed.")



def addKSimilarityclkstrdataset(clkstrdataset):
    
    input_file = clkstrdataset
    temp_file = 'sim'+clkstrdataset
    # Open the input and temporary TSV files
    with open(input_file, 'r', newline='') as input_tsv, open(temp_file, 'w', newline='', encoding='utf-8') as temp_tsv:
        # Create TSV reader and writer
        tsv_reader = csv.reader(input_tsv, delimiter=';')
        tsv_writer = csv.writer(temp_tsv, delimiter=';')
        
        # Write header to the temporary TSV file
        header = ['Entity1', 'Entity2', 'ClickstreamEnt1Ent2', 'PopularityEnt1','PopularityEnt2', 
                  'PopularityDiff', 'PopularitySum', 'CosineSimilarityEnt1Ent2', 'DBpediaSimilarityEnt1Ent2', 'DBpediaRelatednessEnt1Ent2', 'PalmaInterestingnessEnt1Ent2']
        tsv_writer.writerow(header)

        next(tsv_reader)
        # Process each row in the dataset
        for row in tsv_reader:
            # with this loop we bring back the clickstream column as an int
            #row[2] = int(row[2])
            
            ent1, ent2 = row[:2]
         
            #corpussim = pal.distrSemSimilarity(ent1,ent2)
            #corpussim = ""
            cossim = pal.CosineSimilarity("en",ent1,ent2)
            #wupsim = pal.wu_palmer_similarity(ent1,ent2)
            #pathsim = pal.path_similarity(ent1,ent2)
            dbsim = pal.fDBpediaSimilarity(ent1,ent2)
            dbrel = pal.fDBpediaRelatedness(ent1,ent2)
 
            row += [cossim, dbsim, dbrel]
            try:
                tsv_writer.writerow(row)
            except Exception as e:
                continue
            
        # Hereby we add to the next two columns popdiff and popsum
             

def addInterestingnessclkstrdataset(clkstrdataset):
    input_file = clkstrdataset
    temp_file = 'int'+clkstrdataset
    # Open the input and temporary TSV files
    with open(input_file, 'r', newline='', encoding='utf-8') as input_tsv, open(temp_file, 'w', newline='', encoding='utf-8') as temp_tsv:
        # Create TSV reader and writer
        tsv_reader = csv.reader(input_tsv, delimiter=';')
        tsv_writer = csv.writer(temp_tsv, delimiter=';')
        # Write header to the temporary TSV file
        header = ['Entity1', 'Entity2', 'ClickstreamEnt1Ent2', 'PopularityEnt1','PopularityEnt2', 
                  'PopularityDiff', 'PopularitySum', 'CosineSimilarityEnt1Ent2', 'DBpediaSimilarityEnt1Ent2', 'DBpediaRelatednessEnt1Ent2', 'PalmaInterestingnessEnt1Ent2']
        tsv_writer.writerow(header)

        next(tsv_reader)

        # Process each row in the dataset
        for row in tsv_reader:
            # with this loop we bring back the clickstream column as an int
            #row[2] = int(row[2])
            
            ent1, ent2 = row[:2]
            pop = math.log(float(row[6]) + float(row[5])) + 1
            csim = math.log((float(row[7]) + float(row[8]))/2)
            ksim = math.log(float(row[9])+0.1) 
            palmint = (pop * abs(csim - ksim))
            #pal.palma_interestingness(ent1, ent2)
            if palmint == 1:
                continue
            else:
                palmint = round(palmint/math.log10(int(row[2])),2)
                row += [palmint]

                try:
                    tsv_writer.writerow(row)
                except Exception as e:
                    continue


def addFeaturesclkstrdataset(clkstrdataset):
     # Assuming your dataset is stored in a TSV file named 'your_dataset.tsv'
    input_file = clkstrdataset
    temp_file = 'temp_datasetfeatfinal.tsv'

    # Open the input and temporary TSV files
    with open(input_file, 'r', newline='') as input_tsv, open(temp_file, 'w', newline='', encoding='utf-8') as temp_tsv:
        # Create TSV reader and writer
        tsv_reader = csv.reader(input_tsv, delimiter=';')
        tsv_writer = csv.writer(temp_tsv, delimiter=';')
        next(tsv_reader)

        # Process each row in the dataset
        for row in tsv_reader:
            # with this loop we bring back the clickstream column as an int
            #row[2] = int(row[2])
            
            ent1, ent2 = row[:2]
            #kw1 = ""
            #kw1 += pal.SematchFeatures2(ent1)
            #kw2 = ""
            #kw2 += pal.SematchFeatures2(ent2)
            #feat1 = pal.WikifierFeatures(ent1)
            #feat1 += pal.SematchFeatures(ent1)
            #feat2 = pal.WikifierFeatures(ent2)
            #feat2 += pal.SematchFeatures(ent1)
            syns1 = pal.get_babelnet_synset(ent2)
            syns2 = pal.get_babelnet_synset(ent2)
            
            row += [syns1, syns2]

            tsv_writer.writerow(row)
            
            # Hereby we add to the next two columns popdiff and popsum

