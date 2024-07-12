# Import all the needed packages, modules, and classes
from llama_index.llms.ollama import Ollama
from pathlib import Path
import qdrant_client
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core import (VectorStoreIndex,
                              StorageContext,
                              SimpleDirectoryReader,
                              Settings,
                              load_index_from_storage)
from llama_index.readers.file import HTMLTagReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Initialize the LLM
llm = Ollama(model="llama3")
Settings.llm = llm
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model
Settings.chunk_size = 1024
Settings.context_window = 3800

# Check if the storage exists
PERSIST_DIR = "./qdrant_data"
if not Path(PERSIST_DIR).is_dir():
    # load the HTML files
    parser = HTMLTagReader()
    file_extractor = {".html": parser}
    documents = SimpleDirectoryReader("./Bioc_contribution_downloads/",
                                  file_extractor=file_extractor).load_data()
    #Create an index to embed the documents and store them in the vector store
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    #load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

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