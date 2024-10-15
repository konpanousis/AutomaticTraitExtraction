
# Fully automatic extraction of morphological traits from the Web: Utopia or reality?

This is the code implementation for the paper "Fully automatic extraction of morphological traits from the Web: Utopia or reality?", currently under review. 

**Premise**: Plant morphological traits, their observable characteristics, are fundamental to understand the role played by each species within their ecosystem.
However, compiling trait information for even a moderate number of species is a demanding task that may take experts years to accomplish.
At the same time, massive amounts of information about species descriptions is available online in the form of text, although the lack of structure makes this source of data impossible to use at scale.

**Method**: To overcome this, we propose to  leverage recent advances in Large Language Models (LLMs) and devise a mechanism for gathering and processing information on plant traits in the form of unstructured textual descriptions, without manual curation.

**Results**: We evaluate our approach by replicating three manually created species-trait matrices. Our method managed to find values for over half of all species-trait pairs, with an F1-score of over 75%.

**Discussion**:
Our results suggest that large-scale creation of structured trait databases from unstructured online text is currently feasible thanks to the information extraction capabilities of LLMs, being limited by the availability of textual descriptions covering all the traits of interest.

**Obtained Description Sentences**:
In the folder "Descriptions" we include all the obtained descriptive sentences, i.e., sentences that were classified as descriptive by the Description Classifier, for each dataset.

**Required Packages and API keys**:
-----
All the required packages for running the provided notebook are given in the respective requirements.txt file. Install pip (if not installed) and perform the following command:

 ``` pip install -r requirements.txt ```

As mentioned in the notebook itself, an appropriate .env file should be present in the directory where you are running the notebook from. This should include the API keys for your LLM, e.g., MISTRAL, the GOOGLE API key and the Google Custom Search Engine id. An example .env file looks like the following:

```
MISTRAL_API_KEY=YourMistralApiKeyHere
GOOGLE_API=YourGoogleApiKeyHere
GOOGLE_CSE=YourGoogleCustomSearchEngineIDHere
```

With these steps, the configuration is complete. You can now open the **Query_and_Prompt_Species_Traits.ipynb** file and run the notebook.


**Code Structure**
-----
```
 .
 Query_and_Prompt_Species_Traits.ipynb # A jupyter notebook with the full pipeline for querying the search api for certain species, classifying the text and prompting the LLM.
 aggregate_traits.py                   # A script for post processing the obtained results. Writes a summary of the traits in a summary.csv file.
 requirement.txt                       # A file containing all the python packages necessary for running the code.
 .env                                  # This file needs to be created and it should contain all the necessary API keys and CSE ids for each component.
 ├── Descriptions                      # Folder containing the obtained descriptive sentences for each different dataset
     ├── carribean_with_url.zip        # The obtained descriptive sentences for the Caribbean dataset.
     ├── palms_with_url.zip            # The obtained descriptive sentences for the Palms dataset.
     ├── wafrica_with_url.zip          # The obtained descriptive sentences for the West Africa dataset.
 ├── DescriptionClassification         # Scripts pertaining to training the Description Classifier.
     ├── data                          # Scripts for data collection and curation.
     ├── utils                         # Misc Scripts.
     ├── DistilBERT_Train.ipynb        # Sciript to train the BERT Description Classifier.
     ├── predict_model.py              # Load and predict descriptions.
     ├── web_crawler.py                # Data crawling and curation.
 ├── LLMs                              # Script pertaining to data collection and LLM querying.
    ├── Data                           # Contains data crawling, cleaning scripts and other data files.
    ├── PostProcessing                 # Post Processing Scripts to collect the results.
    ├── Results                        # Results for the individual datasets.
    ├── SurveyResults                  # Results for the Surveys.
    ├── MistralPrompt_Surveys.ipynb    # Prompt for the LLM for each dataset.
    └── Mistral_Prompting.ipynb        # Prompt for the LLM for the surveys.
├── models                             # Folder containing the trained description classifier weights

```
