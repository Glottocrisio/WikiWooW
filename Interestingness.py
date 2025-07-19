#If a knowledge graph is in input, this module travels it and attaches to every node its popularity value
#and to every link the interestingness value calculated as in Text2Story2024

# The following function are in almost their entirety
#  copied, adapted or inspired from Wikifier and Sematch python libraries.

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

import requests
import pandas as pd
import matplotlib.pyplot as plt
import tools as t
import urllib.parse, urllib.request, json
from sematch.semantic.similarity import WordNetSimilarity
from sematch.semantic.similarity import YagoTypeSimilarity
from sematch.semantic.graph import DBpediaDataTransform, Taxonomy
from sematch.semantic.similarity import ConceptSimilarity
from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
from huggingface_hub import hf_hub_download
from sematch.semantic.graph import SimGraph
from sematch.semantic.similarity import WordNetSimilarity
from sematch.nlp import Extraction, word_process
from sematch.semantic.sparql import EntityFeatures
from collections import Counter
import nltk
import urllib.parse, urllib.request, json
import sys, io
import math
import numpy as np
from gensim.scripts.glove2word2vec import glove2word2vec
from sematch.nlp import Extraction, word_process
from collections import Counter
from sematch.nlp import RAKE
from sematch.semantic.similarity import EntitySimilarity


# # Load the pre-trained GloVe model (make sure to have the correct path):
# glove_model_path = 'glove.6B.100d.txt'
# word2vec_output_path = 'glove.6B.100d.word2vec'
# enwiki = 'enwiki_20180420_nolg_100d.txt'
# glove2word2vec(glove_model_path, word2vec_output_path)
# glove_model = KeyedVectors.load_word2vec_format(enwiki, binary=False)


#General Wikifier function
def CallWikifier(text, lang="en", threshold=0.6):
    # Prepare the URL.
    data = urllib.parse.urlencode([
        ("text", text), ("lang", lang),
        ("userKey", "hssqbqsagwnptjnqfdsnbmqqcotnud"),
        ("pageRankSqThreshold", "%g" % threshold), ("applyPageRankSqThreshold", "false"),
        ("nTopDfValuesToIgnore", "200"), ("nWordsToIgnoreFromList", "200"),
        ("wikiDataClasses", "true"), ("wikiDataClassIds", "true"),
        ("support", "true"), ("ranges", "true"), ("minLinkFrequency", "2"),
        ("includeCosines", "true"), ("maxMentionEntropy", "3")
        ])
    url = "http://www.wikifier.org/annotate-article"
    # Call the Wikifier and read the response.
    try:
        req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
        with urllib.request.urlopen(req, timeout = 60) as f:
            response = f.read()
            response = json.loads(response.decode("utf8"))
        return response
        # Output the annotations.
        # for rangeg in response["ranges"]:
        #     print(rangeg)
    except Exception as e:
        pass
        
def WikifierPageRank(entity, lang="en", threshold=0.6):
        """
        Method 1: Using Wikifier API (your existing approach)
        """
        data = urllib.parse.urlencode([
            ("text", entity.replace("_"," ")), ("lang", lang),
            ("userKey", "hssqbqsagwnptjnqfdsnbmqqcotnud"),
            ("pageRankSqThreshold", "%g" % threshold), 
            ("applyPageRankSqThreshold", "false"),
            ("nTopDfValuesToIgnore", "200"), 
            ("nWordsToIgnoreFromList", "200"),
            ("wikiDataClasses", "true"), 
            ("wikiDataClassIds", "true"),
            ("support", "true"), 
            ("ranges", "true"), 
            ("minLinkFrequency", "2"),
            ("includeCosines", "true"), 
            ("maxMentionEntropy", "3")
        ])
        
        url = "http://www.wikifier.org/annotate-article"
        
        try:
            req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
            with urllib.request.urlopen(req, timeout=60) as f:
                response = f.read()
                response = json.loads(response.decode("utf8"))
            
            # Extract PageRank from response
            if 'annotations' in response and len(response['annotations']) > 0:
                return response['annotations'][0].get('pageRank', 0)
            elif 'ranges' in response and len(response['ranges']) > 0:
                return response['ranges'][0].get('pageRank', 0)
            else:
                return 0
                
        except Exception as e:
            print(f"Wikifier API error for {entity}: {e}")
            return 0.1
    
# Extract the PageRank Centrality from Wikifier
def WikifierPageRank2(entity):
    try:
        wikiresp = CallWikifier(entity)
        
        # Helper function to safely extract and validate PageRank
        def get_valid_pagerank(item, location="unknown"):
            if isinstance(item, dict) and 'pageRank' in item:
                pr = item['pageRank']
                if pr is None:
                    print(f"PageRank is None for entity '{entity}' at {location}")
                    return None
                try:
                    float_pr = float(pr)
                    if float_pr < 0:  # Additional validation
                        print(f"Negative PageRank ({float_pr}) for entity '{entity}' at {location}")
                        return None
                    return float_pr
                except (ValueError, TypeError) as e:
                    print(f"Cannot convert PageRank '{pr}' to float for entity '{entity}' at {location}: {e}")
                    return None
            return None
        
        # First try: Check annotations
        if wikiresp.get('annotations'):
            for i, annotation in enumerate(wikiresp['annotations']):
                pr = get_valid_pagerank(annotation, f"annotations[{i}]")
                if pr is not None:
                    return round(pr, 2) + 1
        
        # Second try: Check ranges
        for i, range_item in enumerate(wikiresp.get('ranges', [])):
            # Try range directly
            pr = get_valid_pagerank(range_item, f"ranges[{i}]")
            if pr is not None:
                return round(pr, 2) + 1
            
            # Try candidates within ranges
            for j, candidate in enumerate(range_item.get('candidates', [])):
                pr = get_valid_pagerank(candidate, f"ranges[{i}].candidates[{j}]")
                if pr is not None:
                    return round(pr, 2) + 1
        
        # If no valid PageRank found anywhere, return default
        print(f"No valid PageRank found for entity '{entity}', using default value 1")
        return 1
        
    except Exception as e:
        print(f"Error in WikifierPageRank for entity '{entity}': {e}")
        return 1

def debug_wikifier_response(entity):
    """
    Debug function to inspect the actual structure of Wikifier response
    """
    try:
        wikiresp = CallWikifier(entity)
        print(f"\n=== Wikifier Response Structure for '{entity}' ===")
        print(f"Top-level keys: {list(wikiresp.keys())}")
        
        # Check annotations structure
        if 'annotations' in wikiresp:
            print(f"\nAnnotations count: {len(wikiresp['annotations'])}")
            if wikiresp['annotations']:
                print("First annotation keys:", list(wikiresp['annotations'][0].keys()))
                print("First annotation sample:", json.dumps(wikiresp['annotations'][0], indent=2)[:500])
        
        # Check ranges structure
        if 'ranges' in wikiresp:
            print(f"\nRanges count: {len(wikiresp['ranges'])}")
            if wikiresp['ranges']:
                print("First range keys:", list(wikiresp['ranges'][0].keys()))
                
        return wikiresp
    except Exception as e:
        print(f"Error debugging Wikifier response: {e}")
        return None


#WikifierPageRank("Bullet_(1996_film)")

## Extract the incoming clickstream for a single entity (monthly)

def singleclickstream(entity):
    headers = {'User-Agent': 'ISAAKx/1.0 (your@email.com)'}
    url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/'+str(entity).replace("'","").replace(" ","")+'/monthly/20241101/20241130'

    resp = requests.get(url, headers=headers)
    if resp.status_code == 200:
        data = resp.json()
        #print(data['items'][0]['views'])
        return int(data['items'][0]['views'])
    else:
        #print(f"Error: {resp.status_code}")
        #print(resp.text)
        return 1
    
    
def doubleclickstream(entity1, entity2):
    pass

## Extract features from entity using DBpedia (Sematch)

def SematchFeatures(entity):
    ent = EntityFeatures().features('http://dbpedia.org/page/'+str(entity).replace("'","").replace(" ",""))
    labels = Extraction().extract_nouns(ent['abstract'])
    for cat in ent['category']:
        labels.append(cat.replace("http://dbpedia.org/page/Category:",""))
    for topic in ent['type']:
        labels.append(topic.replace("http://dbpedia.org/class/yago/","")[:9])
    #remove duplicates
    labels = list(set(labels))
    print(labels)
    return labels

def SematchYagoFeatures(entity):
    ent = EntityFeatures().features('http://dbpedia.org/page/'+str(entity).replace("'","").replace(" ",""))
    labels = []
    for topic in ent['type']:
        if topic[:30] == "http://dbpedia.org/class/yago/":
            labels.append(topic)
    print(labels)
    return labels
 

def SematchFeatures2(entity):
    ent_features = EntityFeatures().features('http://dbpedia.org/page/'+str(entity).replace("'","").replace(" ",""))
    words = Extraction().extract_nouns(ent_features['abstract'])
    words = word_process(words)
    wns = WordNetSimilarity()
    word_graph = SimGraph(words, wns.word_similarity)
    word_scores = word_graph.page_rank()
    words, scores =zip(*Counter(word_scores).most_common(10))
    words = list(set(list(words)))
    print(words)
    return words

#SematchFeatures2("Pop_art")


#  RAKE algorithm to extract keywords from abstract.
def RAKEfeatures(entity):
    try:
        upm = EntityFeatures()
        upm= upm.features('http://dbpedia.org/page/'+str(entity).replace("'","").replace(" ",""))
        rake = RAKE()
        keywords = rake.extract(upm['abstract'])
        print(keywords)
        return keywords
    except Exception as e:
        pass

#RAKEfeatures("Opel_Corsa")

##Extract features from entity using Wikidata Classes enlabels via Wikifier

def WikifierFeatures(entity):
    wikiresp = CallWikifier(entity)
    features_list = []
    i = 0
    try:
        while i< len(wikiresp['annotations'][0]['wikiDataClasses']):
            label = wikiresp['annotations'][0]['wikiDataClasses'][i]['enLabel']
            features_list.append(label)
            i+=1
        features_list = list(set(features_list))
        print(features_list)
        return(features_list)
    except Exception as e:
        pass

 
def get_babelnet_synset(entity):
    base_url = f'https://babelnet.io/v9/getSynsetIds?lemma={entity}&searchLang=EN&key=3b5675dd-da75-4c2c-b727-c3edf92e4635'
    try:
        response = requests.get(base_url)
        if response.status_code == 200:
            data = response.json()
            bsynset = []
            for datum in data:
                bsynset.append(datum['id'])
            return bsynset
        else:
            print(f"Error: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")


#synset_data = get_babelnet_synset("3b5675dd-da75-4c2c-b727-c3edf92e4635", "Aristotle")
#print(synset_data)


## Functions extracting popularity (modelled as combination of clickstream and centrality)
def entpopularity(entity):
    if singleclickstream(entity) >= 1 and WikifierPageRank(entity) >= 1:
        pop = singleclickstream(entity)/WikifierPageRank(entity)
        return round(pop,2)
    #elif singleclickstream(entity) > 100:
    #    pop = round(singleclickstream(entity)/3, 2)
    #    return pop
    else:
        return 1
    
#entpopularity('Brama_(fish)')

def couplepopularity(ent1, ent2, sum = True):
    if sum:
        cplpop = entpopularity(ent1) + entpopularity(ent2)
    else:
        cplpop = entpopularity(ent1) - entpopularity(ent2)
    return round(cplpop,2)
        
    

## Functions extracting Similarity according to different approaches
## Knowledge-based similarity



#CosineSimilarity("en", "Alexander", "Bucephalus")

# Semantic similarity of Yago classes
def YagoSimilarity(ent1, ent2):
    yago_sim = YagoTypeSimilarity()
    #yagoent1 = SematchYagoFeatures(ent1)
    #yagoent2 = SematchYagoFeatures(ent2)
    yagoent1 = "http://dbpedia.org/class/yago/"+str(ent1).replace("'","").replace(" ","")
    yagoent2 = "http://dbpedia.org/class/yago/"+str(ent2).replace("'","").replace(" ","")
    sim = yago_sim.yago_similarity(yagoent1, yagoent2, 'wpath') 
    print(sim)
    return round(sim,2)

#print("Yagosim:"+ str(YagoSimilarity("Alexander_the_Great", "Bucephalus")))

# Semantic similarity of DBpedia entities
def fDBpediaSimilarity(ent1, ent2):
    try:
        sim = EntitySimilarity()
        similarity = round(sim.similarity('http://dbpedia.org/resource/'+str(ent1).replace("'","").replace(" ",""),'http://dbpedia.org/resource/'+str(ent2).replace("'","").replace(" ","")), 2)
        return similarity
    except Exception as e:
        return 0.01
#print("fDBpediaSimilarity"+ str(fDBpediaSimilarity("Alexander_the_Great", "Bucephalus")))

# Semantic relatedness of DBpedia entities
def fDBpediaRelatedness(ent1, ent2):
    try:
        sim = EntitySimilarity()
        relatedness = round(sim.relatedness('http://dbpedia.org/resource/'+str(ent1).replace("'","").replace(" ",""),'http://dbpedia.org/resource/'+str(ent2).replace("'","").replace(" ","")),2)
        return relatedness
    except Exception as e:
        return 0.01

#print("fDBpediaRelatedness"+ str(fDBpediaRelatedness("Alexander_the_Great", "Bucephalus")))

def fWordNetSimilarity(word1, word2):
    try:
        wns = WordNetSimilarity()
        # Computing English word similarity using Li method
        li_ws = wns.word_similarity(word1, word2, 'li')
        return round(li_ws,2)
    except Exception as e:
        return 0.01

    

##Corpus-based similarity (working also on multi words entities!)

# def distrSemSimilarity(wvec1,wvec2):

#     # Tokenize and preprocess entities
#     tokens1 = wvec1.lower().replace("_",",")
#     tokens1 = t.Utils.cleansing(tokens1)
#     #tokens1 = tokens1.split()
    
#     tokens2 = wvec2.lower().replace("_",",")
#     tokens2 = t.Utils.cleansing(tokens2)
#     #tokens2 = tokens2.split()
#     # Get word vectors for each token
#     vectors1 = [glove_model[word] for word in tokens1 if word in glove_model]
#     vectors2 = [glove_model[word] for word in tokens2 if word in glove_model]

#     # Calculate the average vector for each entity
#     avg_vector1 = np.mean(vectors1, axis=0) if vectors1 else None
#     avg_vector2 = np.mean(vectors2, axis=0) if vectors2 else None

#     # Check if both entities have valid vectors
#     if avg_vector1 is not None and avg_vector2 is not None:
#         # Compute cosine similarity between the average vectors
#         similarity = np.dot(avg_vector1, avg_vector2) / (np.linalg.norm(avg_vector1) * np.linalg.norm(avg_vector2))
#         return round(similarity,2)
#     else:
#         return None

# Example usage
# ent1 = "Alexander_the_Great"
# ent2 = "Bucephalus"
# print(distrSemSimilarity(ent1, ent2))


#Cosine Similarity (Wikifier, cosTfIdfVec)

def CosineSimilarity(lang, title1, title2):
    # Prepare the URL.
    title1 = title1.replace(" ","").replace("_"," ").replace("'","")
    title2 = title2.replace(" ","").replace("_"," ").replace("'","")
    data = urllib.parse.urlencode([("lang", lang),
        ("title1", title1), ("title2", title2)])
    url = "http://www.wikifier.org/get-cosine-similarity?" + data
    # Call the Wikifier and read the response.
    with urllib.request.urlopen(url, timeout = 60) as f:
        response = f.read()
        response = json.loads(response.decode("utf8"))
    # Return the cosine similarity between the TF-IDF vectors.
        try :
            #print(response["cosTfIdfVec"])
            return round(response["cosTfIdfVec"], 2)
        except Exception as e:
            return 0.01

def overview_compute_similarity(entity1, entity2):
    # Create EntitySimilarity and WordNetSimilarity objects
    entity_similarity = EntitySimilarity()
    wordnet_similarity = WordNetSimilarity()

    # Compute semantic similarity
    path_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'path')
    wu_palmer_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'wup')
    jiang_conrath_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'jcn')

    resnik_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'res')
    lin_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'lin')
    
    # Use the EntitySimilarity object to compute entity similarity
    entity_sim = entity_similarity.similarity(entity1,entity2)

    # # Print results
    # print(f"Path Similarity: {path_similarity}")
    # print(f"Wu-Palmer Similarity: {wu_palmer_similarity}")
    # print(f"Jiang-Conrath Similarity: {jiang_conrath_similarity}")
    # print(f"Resnik Similarity: {resnik_similarity}")
    # print(f"Lin Similarity: {lin_similarity}")
    # print(f"Entity Similarity: {entity_sim}")


def path_similarity(entity1, entity2):
    try:
        wordnet_similarity = WordNetSimilarity()
        path_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'path')
        return round(path_similarity,3)
    except Exception as e:
        return 0

def wu_palmer_similarity(entity1, entity2):
    try:
        wordnet_similarity = WordNetSimilarity()
        wu_palmer_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'wup')
        return round(wu_palmer_similarity,3)
    except Exception as e:
        return 0

def entity_similarity(entity1, entity2):
    try:
        entity_similarity = EntitySimilarity()
        entity_sim = entity_similarity.similarity(entity1, entity2)
        return round(entity_sim,3)
    except Exception as e:
        return 0.01

# # Example usage
# entity1 = "Alexander_the_Great"
# entity2 = "Anubis"


# print("entity_similarity"+str(entity_similarity(entity1, entity2)))
# print("path_similarity"+str(path_similarity(entity1, entity2)))
# print("wu_palmer_similarity"+str(wu_palmer_similarity(entity1, entity2)))



# sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding = "utf8",
#                               errors = "ignore", line_buffering = True)
# NeighGraph("sl", "Ljubljana", 0, 1)

# To compute the similarity of two features arrays I have thought of a possible heuristics
# implemented in the following:

# a) Select the top 6 common features of each entity;
# b) Pair them to enhance similarity values on couplets;
# c) Perform the Wordnet similarity for each couplet
# d) Average.


#palmainterestingness(e1-e2) = (ln(popsum) - abs(popdif)) *
# ln (|cosinsimilaritylabelse1labelsee2 − yagosimilaritye1e2|)

def palma_interestingness(ent1, ent2):
    try:
        pop = math.log(couplepopularity(ent1, ent2) + abs(couplepopularity(ent1, ent2, False))) + 1
        csim = math.log(abs((CosineSimilarity("en", ent1, ent2)+fDBpediaSimilarity(ent1, ent2))/2))
        ksim = math.log(fDBpediaRelatedness(ent1,ent2)+0.1) 
        palma = (pop * abs(csim - ksim))
        return round(palma,2)
    except Exception as e:
        return 1

def palma_interestingness2(ent1, ent2):
    """
    Divergence-based interestingness
    - PopularityContrast: Ratio of max to min popularity (normalized)
    - SimilarityAsymmetry: Divergence between semantic and structural similarity
    """
    try:
        # Popularity contrast as ratio with smoothing
        pop1 = couplepopularity(ent1, ent2)
        pop2 = abs(couplepopularity(ent1, ent2, False))
        popularity_contrast = math.log(1 + max(pop1, pop2) / (min(pop1, pop2) + 1))
        
        # Similarity asymmetry as divergence
        semantic_sim = CosineSimilarity("en", ent1, ent2)
        structural_sim = fDBpediaSimilarity(ent1, ent2)
        knowledge_sim = fDBpediaRelatedness(ent1, ent2)
        
        # Compute divergence between different similarity types
        sim_asymmetry = abs(semantic_sim - structural_sim) + abs(semantic_sim - knowledge_sim)
        
        interestingness = popularity_contrast * sim_asymmetry
        return round(interestingness, 2)
    except Exception as e:
        return 1

def palma_interestingness3(ent1, ent2):
    """
    Entropy-based interestingness
    - PopularityContrast: Entropy of popularity distribution (surprisal)
    - SimilarityAsymmetry: Kullback-Leibler divergence proxy
    """
    try:
        # Popularity as entropy (surprisal)
        pop1 = couplepopularity(ent1, ent2)
        pop2 = abs(couplepopularity(ent1, ent2, False))
        total_pop = pop1 + pop2 + 1
        
        # Normalize to probabilities
        p1 = (pop1 + 0.5) / total_pop
        p2 = (pop2 + 0.5) / total_pop
        
        # Shannon entropy
        popularity_contrast = -p1 * math.log(p1) - p2 * math.log(p2)
        
        # Similarity asymmetry as KL divergence proxy
        cosine_sim = CosineSimilarity("en", ent1, ent2) + 0.1
        dbpedia_sim = fDBpediaSimilarity(ent1, ent2) + 0.1
        relatedness = fDBpediaRelatedness(ent1, ent2) + 0.1
        
        # Normalize similarities to [0,1]
        avg_sim = (cosine_sim + dbpedia_sim + relatedness) / 3
        sim_asymmetry = abs(cosine_sim * math.log(cosine_sim / avg_sim)) + \
                       abs(dbpedia_sim * math.log(dbpedia_sim / avg_sim))
        
        interestingness = popularity_contrast * sim_asymmetry
        return round(interestingness, 2)
    except Exception as e:
        return 1

def palma_interestingness4(ent1, ent2):
    """
    Harmonic mean-based interestingness
    - PopularityContrast: Harmonic mean with inverse for contrast
    - SimilarityAsymmetry: Weighted variance of similarity measures
    """
    try:
        # Popularity contrast using harmonic mean
        pop1 = couplepopularity(ent1, ent2) + 1
        pop2 = abs(couplepopularity(ent1, ent2, False)) + 1
        
        # Harmonic mean with its inverse for contrast
        harmonic_mean = 2 / (1/pop1 + 1/pop2)
        arithmetic_mean = (pop1 + pop2) / 2
        popularity_contrast = math.log(arithmetic_mean / harmonic_mean + 1)
        
        # Similarity asymmetry as weighted variance
        cosine_sim = CosineSimilarity("en", ent1, ent2)
        dbpedia_sim = fDBpediaSimilarity(ent1, ent2)
        relatedness = fDBpediaRelatedness(ent1, ent2)
        
        # Compute variance with weights
        sims = [cosine_sim, dbpedia_sim, relatedness]
        mean_sim = sum(sims) / len(sims)
        
        # Weighted variance (giving more weight to outliers)
        sim_asymmetry = sum([(s - mean_sim)**2 * (1 + abs(s - mean_sim)) for s in sims]) / len(sims)
        
        interestingness = popularity_contrast * math.sqrt(sim_asymmetry)
        return round(interestingness, 2)
    except Exception as e:
        return 1

def palma_interestingness5(ent1, ent2):
    """
    Geometric mean-based interestingness
    - PopularityContrast: Geometric dispersion measure
    - SimilarityAsymmetry: Circular variance in similarity space
    """
    try:
        # Popularity contrast as geometric dispersion
        pop1 = couplepopularity(ent1, ent2) + 1
        pop2 = abs(couplepopularity(ent1, ent2, False)) + 1
        
        # Geometric mean and dispersion
        geometric_mean = math.sqrt(pop1 * pop2)
        dispersion = max(pop1, pop2) / min(pop1, pop2)
        popularity_contrast = math.log(dispersion) * math.log(geometric_mean)
        
        # Similarity asymmetry as circular variance
        cosine_sim = CosineSimilarity("en", ent1, ent2)
        dbpedia_sim = fDBpediaSimilarity(ent1, ent2)
        relatedness = fDBpediaRelatedness(ent1, ent2)
        
        # Map similarities to angles [0, 2π]
        angle1 = cosine_sim * 2 * math.pi
        angle2 = dbpedia_sim * 2 * math.pi
        angle3 = relatedness * 2 * math.pi
        
        # Compute circular variance
        cos_mean = (math.cos(angle1) + math.cos(angle2) + math.cos(angle3)) / 3
        sin_mean = (math.sin(angle1) + math.sin(angle2) + math.sin(angle3)) / 3
        
        # Circular variance (1 - R, where R is resultant length)
        resultant_length = math.sqrt(cos_mean**2 + sin_mean**2)
        sim_asymmetry = 1 - resultant_length + 0.1  # Add small constant to avoid zero
        
        interestingness = popularity_contrast * sim_asymmetry
        return round(interestingness, 2)
    except Exception as e:
        return 1