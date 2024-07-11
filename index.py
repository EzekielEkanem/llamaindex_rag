# Import all the needed packages, modules, and classes
from llama_index.llms.ollama import Ollama
from pathlib import Path
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import (VectorStoreIndex, 
                              StorageContext, 
                              SimpleDirectoryReader, 
                              Settings)
from llama_index.readers.file import HTMLTagReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# load the HTML files
parser = HTMLTagReader()
file_extractor = {".html": parser}
documents = SimpleDirectoryReader("./Bioc_contribution_downloads/", 
                                  file_extractor=file_extractor).load_data()

# Initialize the vector store
client = qdrant_client.QdrantClient(path="./qdrant_data")
vector_store = QdrantVectorStore(client=client, collection_name="contributions")
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Initialize the LLM
llm = Ollama(model="llama3")
Settings.llm = llm
Settings.embed_model = "local"

#Create an index to embed the documents and store them in the vector store
index = VectorStoreIndex.from_documents(documents, storage_context=storage_context, 
                                        embed_model=embed_model)

# Query the index
query_engine = index.as_query_engine(llm=llm)

response = query_engine.query("What must a Bioconductor package contain  before it can be accepted for submission")

print(response)



# pip install llama-index-embeddings-huggingface
# pip install llama-index-storage 
# pip install llama-index-storage-storage-context
# pip install llama-index-vector-stores-qdrant
# pip install qdrant_client
# pip install llama-index-llms-ollama
# pip install llama-index
# ollama run llama3
# sudo snap install ollama