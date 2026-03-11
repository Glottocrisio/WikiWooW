# Copyright (c) 2025 Palma. All rights reserved.
# Author: Palma.
#
# If a knowledge graph is in input, this module travels it and attaches to every node its popularity value
# and to every link the interestingness value calculated as in Text2Story2024.
#
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
#from gensim.models.word2vec import Word2Vec
#from gensim.models import KeyedVectors
#from huggingface_hub import hf_hub_download
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
#from gensim.scripts.glove2word2vec import glove2word2vec
from sematch.nlp import Extraction, word_process
from collections import Counter
from sematch.nlp import RAKE
# EntitySimilarity from sematch is broken on Python 3 (syntax errors).
# Replaced with self-contained SPARQL implementations below.
# from sematch.semantic.similarity import EntitySimilarity


# ---------------------------------------------------------------------------
# Helper: strip URI prefix if present, normalize to clean entity name.
# Handles both "Giacomo_Casanova" and "http://dbpedia.org/resource/Giacomo_Casanova"
# ---------------------------------------------------------------------------
def _strip_uri(entity):
    """Strip DBpedia URI prefix if present, normalize spaces to underscores."""
    entity = str(entity).strip().replace("'", "")
    for prefix in ['http://dbpedia.org/resource/', 'https://dbpedia.org/resource/',
                    'http://dbpedia.org/page/', 'https://dbpedia.org/page/']:
        if entity.startswith(prefix):
            entity = entity[len(prefix):]
    return entity.replace(" ", "_")


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
        entity = _strip_uri(entity)
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
        entity = _strip_uri(entity)
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
        entity = _strip_uri(entity)
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
    entity = _strip_uri(entity)
    headers = {'User-Agent': 'ISAAKx/1.0 (your@email.com)'}
    url = 'https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/en.wikipedia/all-access/all-agents/'+str(entity)+'/monthly/20241101/20241130'

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
    entity = _strip_uri(entity)
    ent = EntityFeatures().features('http://dbpedia.org/page/'+str(entity))
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
    entity = _strip_uri(entity)
    ent = EntityFeatures().features('http://dbpedia.org/page/'+str(entity))
    labels = []
    for topic in ent['type']:
        if topic[:30] == "http://dbpedia.org/class/yago/":
            labels.append(topic)
    print(labels)
    return labels
 

def SematchFeatures2(entity):
    entity = _strip_uri(entity)
    ent_features = EntityFeatures().features('http://dbpedia.org/page/'+str(entity))
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
        entity = _strip_uri(entity)
        upm = EntityFeatures()
        upm= upm.features('http://dbpedia.org/page/'+str(entity))
        rake = RAKE()
        keywords = rake.extract(upm['abstract'])
        print(keywords)
        return keywords
    except Exception as e:
        pass

#RAKEfeatures("Opel_Corsa")

##Extract features from entity using Wikidata Classes enlabels via Wikifier

def WikifierFeatures(entity):
    entity = _strip_uri(entity)
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
    entity = _strip_uri(entity)
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
    ent1_clean = _strip_uri(ent1)
    ent2_clean = _strip_uri(ent2)
    yagoent1 = "http://dbpedia.org/class/yago/"+str(ent1_clean)
    yagoent2 = "http://dbpedia.org/class/yago/"+str(ent2_clean)
    sim = yago_sim.yago_similarity(yagoent1, yagoent2, 'wpath') 
    print(sim)
    return round(sim,2)

#print("Yagosim:"+ str(YagoSimilarity("Alexander_the_Great", "Bucephalus")))

# =====================================================================
# DBpedia Similarity & Relatedness — self-contained implementations.
#
# sematch's EntitySimilarity is broken on Python 3 (syntax errors in
# sparql.py). These replacements query DBpedia SPARQL directly.
#
# DIFFERENCE BETWEEN THE TWO:
#
#   fDBpediaSimilarity  = TYPE-BASED (intensional)
#       "How similar are these entities by nature?"
#       Compares the YAGO/DBpedia ontology types (classes) of two entities.
#       E.g. both Alexander_the_Great and Bucephalus have types related
#       to "historical figures" / "ancient world" → moderate similarity.
#       Analogous to WordNet similarity but over the DBpedia/YAGO taxonomy.
#       → Like asking: "Are these the same KIND of thing?"
#
#   fDBpediaRelatedness = LINK-BASED (extensional)
#       "How connected are these entities in the knowledge graph?"
#       Counts how many other entities link to BOTH, using Normalized
#       Google Distance (NGD). Two entities that share many neighbors
#       are highly related even if they are completely different types.
#       E.g. Einstein and Relativity share many linking pages → high.
#       → Like asking: "Do these things appear in the same CONTEXT?"
#
#   fWordNetSimilarity  = WORD-LEVEL (lexical)
#       Compares the WORDS themselves (not the entities) via WordNet,
#       a lexical database of English. Uses synsets (synonym sets) and
#       the hypernym/hyponym hierarchy.
#       E.g. "dog" and "cat" are similar (both under "animal").
#       Only works on single common-noun words, NOT multi-word entity
#       names. "Alexander_the_Great" has no WordNet synset.
#       → Like asking: "Are these WORDS related in English?"
#
# In summary:
#   WordNet    = word-level, lexical taxonomy (English dictionary)
#   DBpediaSim = entity-level, ontological taxonomy (knowledge base types)
#   DBpediaRel = entity-level, graph structure (shared connections)
# =====================================================================

import requests as _requests
from functools import lru_cache as _lru_cache

_DBPEDIA_SPARQL = "https://dbpedia.org/sparql"
_SPARQL_HEADERS = {
    "User-Agent": "WikiWooW-Serendipity/2.0 (research; Python)",
    "Accept": "application/sparql-results+json"
}
_SPARQL_TIMEOUT = 30


def _sparql_query(query, endpoint=None):
    """Execute a SPARQL query against DBpedia and return bindings."""
    url = endpoint or _DBPEDIA_SPARQL
    try:
        print(f"[SPARQL] Querying {url}...")
        resp = _requests.get(
            url,
            params={"query": query, "format": "json"},
            headers=_SPARQL_HEADERS,
            timeout=_SPARQL_TIMEOUT
        )
        print(f"[SPARQL] Status: {resp.status_code}")
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
        print(f"[SPARQL] Results: {len(bindings)} bindings")
        return bindings
    except Exception as e:
        print(f"[SPARQL ERROR] endpoint={url} error={type(e).__name__}: {e}")
        return []


# --- fDBpediaSimilarity: type-based (Jaccard over YAGO/DBO types) ---

# --- fDBpediaSimilarity: now uses wikipedia-api (see below) ---


def _get_wikipedia_categories(entity_name):
    """Get Wikipedia categories for an entity using the plain MediaWiki API."""
    name = _strip_uri(entity_name)
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": name,
        "prop": "categories",
        "cllimit": "max",
        "clshow": "!hidden",
        "redirects": "1",
        "format": "json",
    }
    print(f"  [_get_wiki_cats] Fetching categories for '{name}'...")
    try:
        resp = requests.get(url, params=params,
                            headers={"User-Agent": "WikiWooW/2.0 (research)"},
                            timeout=30)
        print(f"  [_get_wiki_cats] HTTP status: {resp.status_code}")
        resp.raise_for_status()
        data = resp.json()
        
        # Check for redirects
        if "query" in data and "redirects" in data["query"]:
            for redir in data["query"]["redirects"]:
                print(f"  [_get_wiki_cats] Redirected: {redir.get('from')} → {redir.get('to')}")
        
        pages = data.get("query", {}).get("pages", {})
        cats = set()
        for page_id, page in pages.items():
            print(f"  [_get_wiki_cats] Page ID: {page_id}, title: {page.get('title', '?')}")
            if page_id == "-1":
                print(f"  [_get_wiki_cats] PAGE NOT FOUND for '{name}'")
                return set()
            for cat in page.get("categories", []):
                cats.add(cat["title"])
        print(f"  [_get_wiki_cats] Found {len(cats)} categories")
        if cats:
            print(f"  [_get_wiki_cats] Sample: {list(cats)[:5]}")
        return cats
    except Exception as e:
        print(f"  [_get_wiki_cats ERROR] {name}: {type(e).__name__}: {e}")
        return set()


def fDBpediaSimilarity(ent1, ent2):
    """
    Type-based semantic similarity using Wikipedia category overlap (Jaccard).
    Uses plain requests to MediaWiki API.
    Accepts both bare names ("Albert_Einstein") and full URIs.
    """
    print(f"\n[fDBpediaSimilarity] CALLED: '{ent1}' vs '{ent2}'")
    print(f"[fDBpediaSimilarity] Stripped: '{_strip_uri(ent1)}' vs '{_strip_uri(ent2)}'")
    try:
        cats1 = _get_wikipedia_categories(ent1)
        cats2 = _get_wikipedia_categories(ent2)

        print(f"[fDBpediaSimilarity] cats1={len(cats1)}, cats2={len(cats2)}")

        if not cats1 or not cats2:
            print(f"[fDBpediaSimilarity] EMPTY categories → returning 0.01")
            return 0.01

        intersection = cats1 & cats2
        union = cats1 | cats2
        jaccard = len(intersection) / len(union) if union else 0.0

        print(f"[fDBpediaSimilarity] intersection={len(intersection)}, union={len(union)}, jaccard={jaccard:.4f}")
        if intersection:
            print(f"[fDBpediaSimilarity] Shared cats sample: {list(intersection)[:5]}")

        result = round(max(jaccard, 0.01), 2)
        print(f"[fDBpediaSimilarity] RETURNING {result}")
        return result
    except Exception as e:
        print(f"[fDBpediaSimilarity EXCEPTION] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 0.01

#print("fDBpediaSimilarity"+ str(fDBpediaSimilarity("Alexander_the_Great", "Bucephalus")))


# --- fDBpediaRelatedness: link-based (Normalized Google Distance) ---

@_lru_cache(maxsize=500)
def _get_entity_links(entity_name):
    """Get entities linked to/from this entity (both directions) via SPARQL."""
    entity_clean = _strip_uri(entity_name)
    uri = "http://dbpedia.org/resource/" + entity_clean.replace("'", "\\'")
    query = """
    SELECT DISTINCT ?linked WHERE {
      {
        <%s> ?p ?linked .
        FILTER(isURI(?linked))
        FILTER(STRSTARTS(STR(?linked), "http://dbpedia.org/resource/"))
        FILTER(!STRSTARTS(STR(?p), "http://dbpedia.org/ontology/wiki"))
      }
      UNION
      {
        ?linked ?p <%s> .
        FILTER(isURI(?linked))
        FILTER(STRSTARTS(STR(?linked), "http://dbpedia.org/resource/"))
        FILTER(!STRSTARTS(STR(?p), "http://dbpedia.org/ontology/wiki"))
      }
    }
    LIMIT 500
    """ % (uri, uri)

    results = _sparql_query(query)
    return frozenset(r["linked"]["value"] for r in results if "linked" in r)


def fDBpediaRelatedness(ent1, ent2):
    """
    Link-based relatedness using Normalized Google Distance (NGD).
    Measures how many knowledge-graph neighbors two entities share.
    Accepts both bare names and full URIs.
    """
    try:
        links1 = _get_entity_links(ent1)
        links2 = _get_entity_links(ent2)

        if not links1 or not links2:
            print(f"[fDBpediaRelatedness] No links found for "
                  f"{'ent1' if not links1 else ''} {'ent2' if not links2 else ''}: "
                  f"{_strip_uri(ent1)}, {_strip_uri(ent2)}")
            return 0.01

        shared = len(links1 & links2)
        if shared == 0:
            return 0.01

        # Normalized Google Distance
        a = len(links1)
        b = len(links2)
        N = 4298433  # approximate total DBpedia entities (same default as sematch)
        numerator = math.log(max(a, b)) - math.log(shared)
        denominator = math.log(N) - math.log(min(a, b))

        if denominator == 0:
            return 0.01

        ngd = numerator / denominator
        # NGD ∈ [0, ∞); convert to similarity ∈ [0, 1]
        relatedness = max(0.0, 1.0 - ngd)
        return round(max(relatedness, 0.01), 2)
    except Exception as e:
        print(f"[fDBpediaRelatedness error] {e}")
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
    # Strip URI prefix if present, then convert underscores to spaces for Wikifier
    title1 = _strip_uri(title1).replace("_", " ")
    title2 = _strip_uri(title2).replace("_", " ")
    # Prepare the URL.
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
    # Create WordNetSimilarity object
    wordnet_similarity = WordNetSimilarity()

    # Compute semantic similarity
    path_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'path')
    wu_palmer_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'wup')
    jiang_conrath_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'jcn')

    resnik_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'res')
    lin_similarity = wordnet_similarity.word_similarity(entity1, entity2, 'lin')
    
    # Use the self-contained fDBpediaSimilarity instead of broken sematch
    entity_sim = fDBpediaSimilarity(entity1, entity2)

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
        # Uses self-contained SPARQL implementation instead of broken sematch
        return fDBpediaSimilarity(entity1, entity2)
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


POP_CEILING = 200000  # approximate upper bound of monthly pageviews in the dataset
POP_THRESHOLD_RAW = 12000  # monthly pageviews above which an entity is "popular"
TAU_P = POP_THRESHOLD_RAW / POP_CEILING  # ≈ 0.06, normalised popularity threshold


def _normalise_popularity(raw_pop, ceiling=POP_CEILING):
    """Normalise raw popularity (pageviews / PageRank) to [0, 1]."""
    return min(max(raw_pop, 0) / ceiling, 1.0)


def palma_interestingness_lukasiewicz(ent1, ent2, tau_p=TAU_P):
    """
    Corpus–Knowledge Asymmetry serendipity model.

    Serendipity = PopularityContrast × |C − K|

    where C is corpus-based similarity (CosineSimilarity) ∈ [0,1]
    and K is knowledge-based relatedness (DBpediaRelatedness) ∈ [0,1].

    The asymmetry |C − K| captures the divergence between how similar
    two entities appear in text versus how structurally related they are
    in the knowledge graph. A large gap signals that one dimension
    reveals a connection the other misses — a hallmark of serendipity.

    Parameters match the existing palma_interestingness* family so that
    this function can be used as a drop-in alternative in the WikiWooW
    pipeline.
    """
    try:
        print(f"\n{'#'*60}")
        print(f"[INTERESTINGNESS6] CALLED with:")
        print(f"  ent1 = '{ent1}'")
        print(f"  ent2 = '{ent2}'")

        # --- Single-entity popularity (raw pageviews / PageRank) ---
        pop1 = entpopularity(_strip_uri(ent1))
        pop2 = entpopularity(_strip_uri(ent2))
        print(f"[INTERESTINGNESS6] pop1={pop1}, pop2={pop2}")

        # --- Popularity Contrast ∈ [0, 1] ---
        # Uses log-ratio: how different are the two popularities?
        # When both are popular and close → low contrast → low serendipity
        # When there's a big gap → high contrast → potentially serendipitous
        pop_sum = pop1 + pop2
        pop_diff = abs(pop1 - pop2)
        if pop_sum > 0:
            popularity_contrast = pop_diff / pop_sum  # ∈ [0, 1]
        else:
            popularity_contrast = 0.0

        # Boost: also factor in overall popularity level (both popular = interesting)
        # log-normalize the sum to [0, 1]
        pop_level = min(math.log(pop_sum + 1) / math.log(POP_CEILING + 1), 1.0)

        # Combine: average of contrast and level
        popularity_score = (popularity_contrast + pop_level) / 2.0
        print(f"[INTERESTINGNESS6] pop_contrast={popularity_contrast:.4f}, pop_level={pop_level:.4f}, popularity_score={popularity_score:.4f}")

        # --- Similarity measures ∈ [0, 1] ---
        print(f"[INTERESTINGNESS6] Calling CosineSimilarity...")
        C = CosineSimilarity("en", ent1, ent2)
        print(f"[INTERESTINGNESS6] C (CosineSimilarity) = {C}")

        print(f"[INTERESTINGNESS6] Calling fDBpediaRelatedness...")
        K = fDBpediaRelatedness(ent1, ent2)
        print(f"[INTERESTINGNESS6] K (DBpediaRelatedness) = {K}")

        # --- Similarity Asymmetry ∈ [0, 1] ---
        similarity_asymmetry = abs(C - K)
        print(f"[INTERESTINGNESS6] |C - K| = |{C} - {K}| = {similarity_asymmetry:.4f}")

        # --- Final score ∈ [0, 1] ---
        # Geometric mean of popularity_score and similarity_asymmetry
        # Geometric mean ensures both components must contribute
        serendipity = math.sqrt(popularity_score * similarity_asymmetry)
        result = round(serendipity, 2)
        print(f"[INTERESTINGNESS6] sqrt({popularity_score:.4f} * {similarity_asymmetry:.4f}) = {serendipity:.4f}")
        print(f"[INTERESTINGNESS6] RETURNING {result}")
        print(f"{'#'*60}")
        return result

    except Exception as e:
        print(f"[INTERESTINGNESS6 EXCEPTION] {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 0.0


# Model 6: Weighted Serendipity (from Appendix E, eq. weighted serendipity)
#
# S(e1,e2) = λ1 · ½(P(e1)+P(e2)) + λ2 · Sim(e1,e2) + λ3 · Click(e1,e2)
#
# Weights are calibrated from Experiment 2 results on popular-popular pairs:
#
#   Empirical findings (section "Visualization and Evaluation of Results"):
#   - High clickstream ANTI-correlates with serendipity for pop-pop pairs
#     → λ3 must be negative, and has the highest absolute weight.
#   - Similarity (both corpus- and knowledge-based) is the second-strongest
#     predictor for pop-pop pairs → λ2 is positive, moderate.
#   - Popularity contributes least for pop-pop (both already popular)
#     → λ1 is the smallest weight.
#
#   Feature importance from cforest (Table RF importances):
#     Clickstream=0.178, DBpSim=0.127, DBpRel=0.121, CosSim=0.118, PopDiff=0.112
#
#   Derived weight ratios (normalised to |λ1|+|λ2|+|λ3|=1):
#     λ1 = 0.15   (popularity baseline)
#     λ2 = 0.35   (similarity contribution)
#     λ3 = -0.50  (clickstream inverse; highest absolute weight)
#
# =============================================================================

# Default weights calibrated on popular-popular experimental results.
#LAMBDA_1 = 0.15   # popularity
#LAMBDA_2 = 0.35   # similarity
#LAMBDA_3 = 0.50  # clickstream (negative: high clickstream → low serendipity)


def palma_interestingness_weighted(ent1, ent2,
                                    lambda1=0.15,
                                    lambda2=0.35,
                                    lambda3=0.50):
    """
    Weighted linear serendipity model.

    S(e1,e2) = λ1 · ½(P(e1) + P(e2))
             + λ2 · Sim(e1, e2)
             + λ3 · Click(e1, e2)

    Default weights are calibrated for popular-popular entity pairs
    based on the experimental results from Experiment 2:
      λ1=0.15 (popularity), λ2=0.35 (similarity), λ3=-0.50 (clickstream).

    All three components are normalised to [0,1] before weighting so that
    the λ values are directly comparable.

    Parameters
    ----------
    ent1, ent2 : str
        DBpedia entity identifiers.
    lambda1, lambda2, lambda3 : float
        Weights for popularity, similarity, and clickstream respectively.
        Can be overridden for sensitivity analysis or alternative calibrations.
    """
    try:
        # --- Single-entity popularity (normalised to [0,1]) ---
        pop_e1_raw = couplepopularity(ent1, ent2)
        pop_e2_raw = abs(couplepopularity(ent1, ent2, False))
        p1 = _normalise_popularity(pop_e1_raw)
        p2 = _normalise_popularity(pop_e2_raw)

        popularity_term = 0.5 * (p1 + p2)

        # --- Similarity (average of corpus- and knowledge-based) ---
        cosine_sim = CosineSimilarity("en", ent1, ent2)       # ∈ [0,1]
        dbpedia_sim = fDBpediaSimilarity(ent1, ent2)           # ∈ [0,1]
        relatedness = fDBpediaRelatedness(ent1, ent2)          # ∈ [0,1]

        # Sim(e1,e2): average of all three similarity measures
        similarity_term = (cosine_sim + dbpedia_sim + relatedness) / 3.0

        # --- Clickstream (normalised to [0,1] via log compression) ---
        # Raw clickstream can range from 0 to millions; log-compress and
        # normalise against an empirical ceiling.
        CLICK_LOG_CEILING = math.log(1_000_000 + 1)  # approx upper bound
        raw_click = couplepopularity(ent1, ent2)  # directed clickstream
        click_normalised = math.log(abs(raw_click) + 1) / CLICK_LOG_CEILING
        click_normalised = min(click_normalised, 1.0)

        # --- Weighted combination ---
        serendipity = (lambda1 * popularity_term
                     + lambda2 * similarity_term
                     + lambda3 * click_normalised)

        return round(serendipity, 2)

    except Exception as e:
        return 1


# =============================================================================
# Convenience: score a triplet <e1, m, e2> using any of the above models.
# =============================================================================

def score_triplet(e1, m, e2, model_fn, aggregation="mean"):
    """
    Score the serendipity of entity triplet <e1, m, e2> by applying a
    couple-level model to both legs and aggregating.

    Parameters
    ----------
    e1, m, e2 : str
        Source entity, intermediary, target entity.
    model_fn : callable
        Any function with signature f(ent1, ent2) -> float.
        E.g. palma_interestingness, palma_interestingness_lukasiewicz, etc.
    aggregation : str
        "mean" : arithmetic mean of both legs (default).
        "min"  : minimum of both legs (path is as serendipitous as its
                 weakest segment).
        "max"  : maximum (optimistic aggregation).

    Returns
    -------
    float
        Aggregated serendipity score for the triplet.
    """
    s1 = model_fn(e1, m)
    s2 = model_fn(m, e2)

    if aggregation == "mean":
        return round((s1 + s2) / 2, 2)
    elif aggregation == "min":
        return round(min(s1, s2), 2)
    elif aggregation == "max":
        return round(max(s1, s2), 2)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")