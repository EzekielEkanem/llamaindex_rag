# Import all functions from index_functions.py 
from index_functions import *

# list of models to be evaluated
models = ["llama3.1:latest", "llama3", "gemma:2b", "gemma:7b"]
# embedding model
embed_model_name = "BAAI/bge-small-en-v1.5"

# get all initialized llms as well as the embedding model used
llms, embed_model = init_llms(models, embed_model_name)

# index the document we want to use as the RAG store (in our case we are indexing
# documents from the bioconductor contributions page https://contributions.bioconductor.org/index.html)
index = index_documents(embed_model)

# location of the package we want ot query
package_query_location = "./EnsemblGenomes"
# the llm we want to use as a judge
judge_llm = "llama3.1:latest"
# the evaluation of each llm
llms_responses = query_llms(package_query_location, llms, index, judge_llm)
# print each llm response
for llm, responses in llms_responses:
    print(llm + "\n")
    for response in responses:
        print(response)