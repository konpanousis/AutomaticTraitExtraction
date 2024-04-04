import spacy
import pickle
from spacy import displacy
nlp = spacy.load('en_core_web_trf')
from bs4 import BeautifulSoup
import requests
from tqdm.notebook import tqdm as tqdm_notebook
import collections
import re
import pandas as pd
import numpy as np
import networkx as nx
from networkx.drawing.nx_agraph import graphviz_layout
import matplotlib.pyplot as plt

def glossary():
    # URL
    URL = 'https://en.wikipedia.org/wiki/Glossary_of_bird_terms'
    # Get the page
    page = requests.get(URL, timeout=5)
    soup = BeautifulSoup(page.content, "lxml", from_encoding="iso-8859-1")   
    # Find embedded glossary
    glossaries = soup.find_all('dt', {'class': 'glossary'})
    parts = [part.text.lower().strip() for part in glossaries]
    # Get additional anchors ("also know as...")
    glossaries_other = soup.find_all('span', {'class': 'anchor'})
    parts_other = [part['id'].lower().strip() for part in glossaries_other]
    # Append and drop duplicates
    parts = list(set((parts + parts_other)))
    # Replace underscore with space
    glossary = [part.replace('_', ' ') for part in parts]

    # Extra
    additions = [
        'legs',
        'beak',
        'head',
        'wingspan',
        'eye',
        'forecrown'
    ]

    glossary += additions
    
    return glossary

def determination(doc):
    return 

def traits(token, doc):
    if token.lemma_.lower() in glossary or token.text.lower() in glossary:
        trait = token.lemma_.lower()
        # Correct traits
        if doc[token.i - 1].lemma_ in compounds:
            trait = doc[token.i - 1: token.i + 1].lemma_.lower()
        elif doc[token.i - 1].dep_ == 'compound':
            trait = doc[token.i - 1: token.i + 1].lemma_.lower()
        return trait

def adjectives(token, doc):
    
    node = None
    # Skip locations
    if token.lemma_ in compounds or token.lemma_ in sex_determ or token.lemma_ in glossary:
        pass
    # Skip 
    elif token.dep_ in ['acl', 'relcl']:
        pass
    elif token.lemma_ in ['have', 'be']:
        pass
    elif token.i < doc[-1].i and doc[token.i + 1].text == '-':
        #print(token)
        pass
    elif token.text == '-':
        pass
    elif token.pos_ == 'VERB':
        # DO NOT LEMMATIZE VERBS
        node = token.text
    elif token.pos_ == 'ADJ':
        if doc[token.i - 1].text == '-':
            node = doc[token.i - 2 : child.i + 1].lemma_ 
        else:
            node = token.lemma_ 
    return node
        
def normal_subject(token, doc):
    
    edge = None
    node = None
    # Get first parent
    parent = next(token.ancestors)
    if parent.lemma_ == 'be':           
        for child in parent.children:
            if child.dep_ == 'acomp':
                edge = 'be'
                if doc[child.i - 1].text == '-':
                    node = doc[child.i - 2 : child.i + 1].text
                else: 
                    node = child.text
    else:
        for child in parent.children:
            if child.dep_ == 'advmod':
                edge = parent.text
                if doc[child.i - 1].text == '-':
                    node = doc[child.i - 2 : child.i + 1].lemma_
                else: 
                    node = child.lemma_
            elif child.dep_ == 'auxpass':
                edge = 'be'
                if doc[child.i - 1].text == '-':
                    node = doc[child.i - 2 : child.i + 1].text
                else: 
                    node = parent.text
    return edge, node

def adjective_subject(token, doc):
    
    edge = None
    node = None
    # Get first parent
    parent = next(token.ancestors)
    if parent.dep_ == 'ROOT' and parent.pos_ == 'ADJ':
        edge = 'be'
        node = parent.lemma_
    return edge, node 
    
def prep_obj(token, doc):
    
    edge = None
    node = None
    parent = next(token.ancestors)
    if parent.dep_ == 'prep':
        parent = next(parent.ancestors)
        if parent.pos_ == 'NOUN':
            edge = 'have'
            node = parent.lemma_
    return edge, node, parent

def clean_list(RDFs):
    
    # Drop empty
    RDFs = [RDF for RDF in RDFs if all(RDF)]
        
    if len(RDFs) <= 1:
        return None
    
    triples = []
    for idx, triple in enumerate(RDFs):
        if idx + 1 == len(RDFs) and triple[0] == 'bird':
            continue        
        if not triple[2]:
            continue
        if triple[0] == 'bird' and triple[2] not in RDFs[idx + 1][0]:
            continue
        if triple[0] == 'bird' and not RDFs[idx + 1][2]:
            continue
        else:
            triples.append(triple)
    return triples