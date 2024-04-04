

def text_cleaner(dirty_text):
    
    import spacy
    import re
    nlp = spacy.load('en_core_web_lg') # NO TRF FOR OOV
    
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
    
    return list(set(sents_clean))
    
    


def DuckDuckGo_Java(query, driver):
    
    from selenium import webdriver    
    from bs4 import BeautifulSoup
    import time
    
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
    
    import requests
    import re
    from bs4 import BeautifulSoup
    
    
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


def web_crawler(species, 
                queries=False, 
                DuckDuckGo=False,
                Bing=False                
               ):
    
    from selenium import webdriver 
    import requests
    from bs4 import BeautifulSoup
    
    search_links = []
    sentence_list = []
    
    if not queries:
        queries = ['description', 
                   'diagnosis', 
                   '', 
                   'attributes', 
                   'captions']
    # Define own queries
    else:
        queries=queries
        
    # Start driver outside of the loop
    if DuckDuckGo:
        driver = webdriver.Safari()
        
    for query in queries:
        # create query
        species_q = species.replace(' ', '+')
        species_q = f'"{species_q}"+{query}'
        if DuckDuckGo:
            search_links += DuckDuckGo_Java(species_q, 
                                            driver=driver)
        if Bing:
            search_links += Bing_HTML(species_q)
         # Skip connection timeout
    if DuckDuckGo:
        driver.close()
    # Drop duplicates
    search_links = list(set(search_links))
    
    if not search_links:
        print('No URLs found')
    
    for URL in search_links:
        # Skip google archives
        if 'google' in URL:
            continue
        # PDF and TXT
        if URL.endswith('txt') or URL.endswith('pdf'):
            continue
        
        try:
            page = requests.get(URL, timeout=2)
            # Skip PDF files for now
            if page.headers['Content-Type'].startswith('application/pdf'):
                continue
            # Soup the result
            soup = BeautifulSoup(page.content, "lxml", from_encoding="iso-8859-1")    
            # Skip Embedded PDF's
            if 'pdf' in soup.title.text.lower():
                continue
            # Check if species exists somewhere within title
            if bool(set(species.split()).intersection(soup.title.text.split())):
                # Get text
                #dirty_text = soup.get_text(". ", strip=True)
                dirty_text = soup.get_text(" ", strip=False).replace('\n', '.')
                # Clean and break into sents
                sentences = text_cleaner(dirty_text)
                # Append
                sentence_list.extend(sentences)
        except:
            continue
                
    return sentence_list
        