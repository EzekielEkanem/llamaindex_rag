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
llm = Ollama(base_url="http://localhost:11430", model="llama3.1", request_timeout=500)
Settings.llm = llm
embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5")
Settings.embed_model = embed_model   
# Settings.chunk_size = 1024
# Settings.context_window = 3800

# Check if the storage exists
PERSIST_DIR = "./qdrant_data"
if not Path(PERSIST_DIR).is_dir():
    # load the HTML files
    parser = HTMLTagReader()
    doc_path = "/home/ubuntu/llamaindex_rag/Bioc_contribution_downloads"
    # file_extractor = {".html": parser}
    documents = SimpleDirectoryReader(input_dir=doc_path,
            required_exts=[".html"], recursive=True).load_data()
    #Create an index to embed the documents and store them in the vector store
    index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, 
                                            show_progress=True)
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    #load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Query the index
question = "What must a Bioconductor package contain  before it can be accepted for submission"
chat_engine_1 = index.as_chat_engine(llm=llm)

response_1 = chat_engine_1.chat(question)
print(f"llm response 1:\n {response_1}")

input_text = f"""
The question was, {question}.
Answer from the llm was {response_1}
Supporting Documents: {response_1.source_nodes}
Task: 
Please, evaluate the correctness, relevance, and completeness of the answer and the 
source nodes provided above. Is the above response correct? what else is missing from 
the response? what can you add, if necessary? Please, meticulously provide these answers
"""

chat_engine_2 = index.as_chat_engine(llm=llm)
response_2 = chat_engine_2.chat(input_text)



# # stream chat
# streaming_response = chat_engine.stream_chat("What must a Bioconductor package contain  before it can be accepted for submission")

# for token in streaming_response.response_gen:
#     print(token, end="")

print(f"llm response 2:\n {response_2}")

revised_text = f"""
The question was, {question}.
The answer you gave was: {response_1}
Feedback from the review: {response_2}
Task:
Please, refine your answer by incorporating this feedback and adding more information where necessary
"""

response_3 = chat_engine_1.chat(revised_text)

print(f"llm response 3:\n {response_3}")