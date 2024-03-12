#This module implements the creation of a Dataset derived from wikipedia clickstream-data
#data dump including all other measures (centrality, popularity, similarity)
import os
import csv
import requests
import pandas as pd
import matplotlib.pyplot as plt
import tools as t
import Palma_Interestingness as pal
import metrics as me
import math

## the output txt file will have this columns titles (in this order):
##entity1 entity2 clickstreame1-e2 pope1 pope2 popei-popee2 (popdiff) pope1+pope2 (popsum)
## e1labelswikiclasses e2labelswikiclasses cosinsimilaritylabelse1labelsee2 yagosimilaritye1e2 #palmainterestingness
 

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
    
    # Assuming your dataset is stored in a TSV file named 'your_dataset.tsv'
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

