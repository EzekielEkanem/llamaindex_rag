# ILLAMAINDEX RAG 

This package utilizes llamaindex to make llms review Bioconductor package submissions according to the guidelines given on the [Bioconductor contributions page](https://contributions.bioconductor.org/). 
It first creates a RAG store for the [Bioconductor contributions page](https://contributions.bioconductor.org/), then it takes in a package and initializes several llms to evaluate the 
R files and vignettes of the package. A llm that serves as the judge llm reviews the evaluation of these llms and gives feedback concerning improvements that can be made to the evaluation. 
Then each llm polishes it evaluation based on the feedback given by the judge. This makes the llm give a more accurate evaluation of the package.

## GETTING STARTED
To run the package the follow installations are necessary:
- [ ] sudo snap install ollama
- [ ] pip install llama-index
- [ ] pip install llama-index-llms-ollama
- [ ] pip install llama-index-storage-storage-context
- [ ] pip install llama-index-storage
- [ ] pip install llama-index-embeddings-huggingface

## Scripts
**llms_review_pipeline.py**
- [ ] Utilizes the functions in index_functions.py to initializes llms and index the documents from the Bioconductor contributions page
- [ ] Uses the initialized llms to review package submissions using the stored vector database derived from the indexed documents
- [ ] Prints the final evaluation of each llm

**index_functions.py**
- [ ] Imports all needed classes, modules, and packages from llamaindex
- [ ] Initializes all the llms to be used for the evaluation
- [ ] Makes a RAG vector store for guidelines on the Bioconductor contributions page
- [ ] Enables each llm to evaluate packages, get feedback of their evaluation, and modify their evaluation based on the feedback given by another llm

To run the package, copy and paste the command below on the command line:
  - `python3 llms_review_pipeline.py`
