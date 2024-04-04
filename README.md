# Fully automatic extraction of morphological traits from the Web: Utopia or reality?

This is the code implementation for the paper "Fully automatic extraction of morphological traits from the Web: Utopia or reality?", currently under review. 

**Premise**: Plant morphological traits, their observable characteristics, are fundamental to understand the role played by each species within their ecosystem.
However, compiling trait information for even a moderate number of species is a demanding task that may take experts years to accomplish.
At the same time, massive amounts of information about species descriptions is available online in the form of text, although the lack of structure makes this source of data impossible to use at scale.

**Method**: To overcome this, we propose to  leverage recent advances in Large Language Models (LLMs) and devise a mechanism for gathering and processing information on plant traits in the form of unstructured textual descriptions, without manual curation.

**Results**: We evaluate our approach by replicating three manually created species-trait matrices. Our method managed to find values for over half of all species-trait pairs, with an F1-score of over 75%.

**Discussion**:
Our results suggest that large-scale creation of structured trait databases from unstructured online text is currently feasible thanks to the information extraction capabilities of LLMs, being limited by the availability of textual descriptions covering all the traits of interest.
