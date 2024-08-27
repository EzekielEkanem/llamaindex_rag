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

files = Path("./EnsemblGenomes/R").glob("**/*.R")
for file in files:
    print (str(file))
    question = f"""
    You are a package reviewer. Your job is to review R files and ensure that the code is 
    written according to the guidelines provided by Bioconductor. Evaluate the R code in 
    this {file} based on the Bioconductor guidelines. Report any deviation found in the R code
    that is different from what is expected as a Bioconductor package (for example, if any 
    line is longer than 80 characters, report it). Suggest improvements to the code where 
    necessary according to the guidelines given by Bioconductor, and cite sources if necesssary.
    If the R file is well written, commend on areas where you think the writter of the file did 
    and excellent job
    """

    # Query the index
    chat_engine_1 = index.as_chat_engine(llm=llm, max_iterations=100)

    response_1_stream = chat_engine_1.stream_chat(question)
    response_1 = ""
    for token in response_1_stream.response_gen:
        response_1 += token
    print(f"llm response 1:\n {response_1}")

    input_text = f"""
    The question was, "{question}".
    Answer from the llm was "{response_1}". Is the above response correct? what else is missing from 
    the response? what can you add, if necessary? Please, meticulously provide these answers
    """

    chat_engine_2 = index.as_chat_engine(llm=llm, max_iterations=100)
    response_2_stream = chat_engine_2.stream_chat(input_text)
    response_2 = ""
    for token in response_2_stream.response_gen:
        response_2 += token
    print(f"llm response 2:\n {response_2}")

    revised_text = f"""
    The question was, "{question}".
    The answer you gave was "{response_1}", and the
    feedback from the review was "{response_2}". 
    Please, refine your answer by incorporating this feedback and adding more information where necessary
    """

    chat_engine_3 = index.as_chat_engine(llm=llm, max_iterations=100)
    response_3_stream = chat_engine_3.stream_chat(revised_text)
    response_3 = ""
    for token in response_3_stream.response_gen:
        response_3 += token

    print(f"llm response 3:\n {response_3}")

# # stream chat
# streaming_response = chat_engine.stream_chat("What must a Bioconductor package contain  before it can be accepted for submission")

# for token in streaming_response.response_gen:
#     print(token, end="")