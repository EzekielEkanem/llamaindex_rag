from llama_index.llms import Ollama

llm = Ollama(model="llama3")

response = llm.complete("What is bioconductor")

print(response)