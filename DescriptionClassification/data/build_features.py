import random
import time
from tqdm import tqdm
import re
from itertools import chain
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pickle
import numpy as np
import torch.nn as nn
import torch
import requests 
from matplotlib import cm
import matplotlib
from bs4 import BeautifulSoup
from selenium import webdriver
from sklearn.metrics import classification_report
import spacy
from spacy import displacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.util import filter_spans
nlp = spacy.load("en_core_web_lg")

from transformers import DistilBertTokenizer, DistilBertModel
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

import sys
sys.path.insert(0, '/src/models/')
from predict_model import SpanPredictor as classify
import predict_model
sim_model = predict_model.load_simBERT()

def random_text_splitter(text):
    
    """
    Random breaks up a text into an X amount of sentences. 
    The output sentences consist of a minimum of 10 sentences.
    """

    # Split text
    words = text.split()
    # Get the amount of words
    word_amount = len(words)
    # Create counter
    remaining_word_amount = word_amount
    # Init list
    parts = []
    # While words remaining
    while remaining_word_amount > 0:
        if len(words) < 10:
            # Add last part if less then 10
            parts[-1] = parts[-1] + ' '.join(words)
            # exit
            remaining_word_amount = 0
        # Generate random int
        randint = random.randint(10, word_amount)
        # Append to list 
        parts.append(' '.join(words[:randint]))
        # Delete previous selection
        words = words[randint:]
        # Update counter
        remaining_word_amount -= randint
        
    return parts

def text_cleaner(dirty_text, per_sent=True):
    
    """
    Cleans the contents of a string object and uses SpaCy to return single sentences.
    """    
    
    regexes = [
        (r'\(\d+.+?Close\n\t\n\)', ''),
        (r'\(.+?\)', ''),
        (r'\[.+?\]', ''),
        (r'\.\.\.', '.'),
        (r'\.\s*\.', '.'),
        (r'-*subsp\.', 'subspecies'),
        (r'-*var\.', 'variation'),
        (r'\s*?…', ''),
        (r'Morphology:\xa0', ''),
        (' +', ' '),
    ]

    # Clean text
    for regex, replace in regexes:
        dirty_text = re.sub(regex, replace, dirty_text)
    # Clean stuff
    text = dirty_text.replace('\r', "")\
                     .replace('\n', "")\
                     .replace('\t', "")\
                     .replace(';', ',')\
                     .strip()
                     #.replace(';', '.')\
                     #.strip()
                     #.encode("ascii", "ignore")\
                     #.decode()\

    #nlp
    doc = nlp(text)
    sents = [i for i in doc.sents]
    
    sents_clean = []
    # Clean non English
    for sentence in sents:
        # Skip short stuff
        if len(sentence) <= 5:
            continue
        # Create ratio
        non_eng = [token.is_oov for token in sentence].count(True)
        # Continue if the ratio is bad (non English jibberisch)
        if non_eng > 0 and non_eng / len(sentence) > .2:
            continue
        # Skip too much rubble
        if non_eng > 8:
            continue
                
        # Skip sentences without upper, results in mostly sentence part without information
        if not sentence[0].is_title: #or not sentence[-1].is_punct:
            continue
            
        # Clean dots
        sentence  = re.sub(r'(\s*\.){1,}', '.', sentence.text)
        
        sents_clean.append(sentence)
    
    sents_clean = list(set(sents_clean))
    
    if per_sent:
        return sents_clean
    else:
        return doc
    

def get_prediction_results(data_values, model,
                           index=-1, 
                           soft_error=False, 
                           beta=0.95):
    
    """
    Uses a dictionary with species names. Undicts the dict and returns
    a precision/recall plot that can be printed. The second value returned
    contains a list with missclassified sentences. Optionally a soft_error can 
    used.
    """

    # init arrays and list
    y_list = np.array([])
    pred_list = np.array([])
    misclass_list = []
    y_onehot = np.array([]).reshape(0, 2) 
    
    # Drop duplicates
    #data_values = list(set(data_values))
    
    # loop over the values of the list
    for (label, sent) in tqdm(data_values):
        #sent_str = sent.text
        sent_str = sent

        # Soft error instead of hard error
        if soft_error:
            pred = classify(sent_str, model=model, pred_values=True)
            pred_np = pred[1][1].numpy().item()
            prediction = pred[0]
            if beta < pred_np or pred_np < 1-beta:
                label = pred[0]
        else:
            # Get prediciton
            prediction = classify(sent_str, model=model)

        # Append for plotting
        #y_onehot = np.vstack([y_onehot, [1, 0]]) if label == 0 else y_onehot = np.vstack([y_onehot, [0, 1]])           
        if label == 0:
            y_onehot = np.vstack([y_onehot, [1, 0]])
        else:
            y_onehot = np.vstack([y_onehot, [0, 1]])

        # Stack horizontally
        pred_list = np.hstack([pred_list, prediction])
        y_list = np.hstack([y_list, label])
        # Append the missclassified sents
        if label != prediction:
            misclass_list.append(tuple([f'real:{label} pred:{prediction}', sent_str]))
    
    # Generate pres/recall            
    report = classification_report(y_list, pred_list) 
    
    return report, misclass_list, y_onehot


def similarity_matrix(sentence_list, model):
    
    """
    Calculates a hidden state array per sententence based on a list of
    sentences.
    """
    
    # Initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentence_list:
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=512,
                                           truncation=True, 
                                           padding='max_length',
                                           return_tensors='pt')
        # Drop the batch dimension
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])
    
    # Reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])
    
    # Get vectors
    hiddenstates = model(**tokens)
    # Sum along first axis
    summed_hs = torch.sum(hiddenstates, 1)
    # Detach
    summed_hs_np = summed_hs.detach().numpy()
    # Get the matrix
    #return cosine_similarity(summed_hs_np, summed_hs_np).round(5)
    return summed_hs_np


def name_cleaner(key, sentence, replacement):
    
    """ 
    Removes the species names from a sentence.
    
    key         : Species name
    sentence    : Sentence to clean
    replacement : Replacement value
    """
    key = key.lower()
    key_list = key.split()
    sentence = sentence.lower()
    
    if len(key_list) == 1:
        # Init regexes
        regexes = [f'{key}',]
    else:    
        # Init regexes
        regexes = [f'{key}',
                   f'{key_list[0][0]}.\s*?{key_list[1]}',
                   f'{key_list[0]}',
                   f'{key_list[1]}',]
    
    # Clean based on regexes
    for regex in regexes:
        sentence = re.sub(regex, replacement, sentence)
        
    return sentence.capitalize()

def extract_subject(sent):
    
    sent = sent.replace('cm', '')\
               .replace('mm', '')
    
    doc = nlp(sent)
    
    # Check for normal subject
    subject = [token for token in doc if token.pos_ == 'NOUN' and token.dep_ == 'nsubj']
    if subject != []:
        return subject[0]
    # Check for ROOT subject
    subject = [token for token in doc if token.pos_ == 'NOUN' and token.dep_ == 'ROOT']
    if subject != []:
        return subject[0]
    # Else return nothing
    return None

def yield_chunks(sentence):
    
    sentence_list = sentence.split(', ')
    cleaned = []
    
    for idx, sent in enumerate(sentence_list):

        # Find subject:
        subject = extract_subject(sent)
        if subject:
            cleaned.append(sent)
        # Find subject of previous subject
        else:
            try:
                subject = extract_subject(cleaned[-1])
                if subject:
                    cleaned.append(f'{subject} {sent}')
                # In case nothing found append 
                else:
                    cleaned.append(f'{sent}')
            except:
                cleaned.append(f'{sent}')
    
    return cleaned

def DuckDuckGo_Java(query, driver):
    
    """
    Uses a safari browser to return search links.
    
    query : query to use
    driver: driver to use for the query (e.g. FireFox, Chrome).
            Needs to be booted before the search
    """
    
    # Init driver
    # driver = webdriver.Safari()
    # Login URL
    driver.get(f'https://duckduckgo.com/?q={query}&t=hy&va=g&ia=web')
    #
    time.sleep(2.5)
    # Read the XML file
    HTML = driver.page_source
    # Species name
    soup = BeautifulSoup(HTML, 'html.parser')

    results = soup.find_all('a', attrs={'class':'result__a'}, href=True)
    # Init list
    links = [result['href'] for result in results if result['href'].startswith('https')]
    
    return links

def Bing_HTML(query):
    
    """
    Queries Bing and returns a URL list.
    """
    
    # Get results
    page = requests.get('https://www.bing.com/search?form=MOZLBR&pc=MOZI&q={0}'.format(query), 
                        headers={'user-agent': 'Descriptor/0.0.1'})
    soup = BeautifulSoup(page.content, 'html.parser')
    # Init list
    links = [] 
    # Clean results
    for i in soup.find_all('a', attrs={'h':re.compile('ID=SERP.+')}, href=True):
        link = i['href']
        if link.startswith('http') and 'microsoft' not in link and 'bing' not in link:
            links.append(link)        
            
    return links


def colorize_prediction(sentence_list, model, tex=False):

    # Get prediction values
    sentence_pred = [classify(sent, model=model, pred_values=True)[1][1].item() for sent in sentence_list]
    # Get color map
    sentence_cmap = matplotlib.cm.BuGn
    # Resample to prevent dark green
    
    template = """  <mark class="entity" style="
    background: {}; 
    padding: 0.4em 0.0em; 
    margin: 0.0em; 
    line-height: 2; 
    border-radius: 0.75em;
    ">{}    
    <span style="
    font-size: 0.8em; 
    font-weight: bold;
    font-color: #538b01;
    font color: #538b01;
    line-height: 1; 
    border-radius: 0.75em;
    text-align: justify;
    text-align-last:center;
    vertical-align: middle;
    margin-left: 0rem">
    </span>\n</mark>"""

    colored_string = ''
    
    # Tex list
    tex_colors = []
    tex_text = []
    HTML = 'HTML'
    
    # Map the values
    normalized_and_mapped = matplotlib.cm.ScalarMappable(cmap=sentence_cmap).to_rgba(sentence_pred)
    # Color overlay the values
    for idx, (sentence, color, prediction) in enumerate(zip(sentence_list, normalized_and_mapped, sentence_pred)):
        
        sentence = f'{sentence} < {prediction:.3f} >'
        color = matplotlib.colors.rgb2hex(color)
        colored_string += template.format(color, sentence)
        
        ## TEX PART
        if tex:
            tex_colors.append(f'\definecolor{{color{idx+1}}}{{{HTML}}}{{{color[1:]}}}')
            tex_text.append(f'\sethlcolor{{color{idx+1}}}\hl{{{sentence}}}')
            
    if tex:
        print('Copy paste this in the .tex file')
        print('\n'.join(tex_colors))
        print('\n'.join(tex_text))
    
    
    #display(HTML(colored_string))
    #output_path = Path("test.html")
    #output_path.open("w", encoding="utf-8").write(colored_string)
    return colored_string