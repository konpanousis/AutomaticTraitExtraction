import random
import time
from tqdm import tqdm
import re
from itertools import chain
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import pickle
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

