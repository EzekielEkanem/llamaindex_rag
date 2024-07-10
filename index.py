# Import all the needed packages, modules, and classes
from llama_index.llms.ollama import Ollama
from pathlib import Path
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.storage.storage_context import StorageContext
from llama_index import (VectorStoreIndex, ServiceContext)
from llama_index.readers.file import HTMLTagReader
from llama_index.core import SimpleDirectoryReader

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
service_context = ServiceContext.from_defaults(llm=llm, embed_model="local")

#Create an index to embed the documents and store them in the vector store
index = VectorStoreIndex.from_documents(documents, service_context=service_context, 
                                        storage_context=storage_context)

# Query the index
query_engine = index.as_query_engine()

response = query_engine.query("What is bioconductor")

print(response)