# Import all the needed packages, modules, and classes
from llama_index.llms.ollama import Ollama
from pathlib import Path
from llama_index.core import (VectorStoreIndex,
                              StorageContext,
                              SimpleDirectoryReader,
                              Settings,
                              load_index_from_storage)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def init_llms(models, embed_model_name, request_timeout=500):
    llms = []
    for model in models:
        if "llama" or "gemma" in model:
            llm = Ollama(model=model, request_timeout=request_timeout)
        llms.append(llm)
    embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name)
    Settings.embed_model = embed_model   
    # Settings.chunk_size = 1024
    # Settings.context_window = 3800
    return llms, embed_model

def index_documents(embed_model, storage_path="./qdrant_data", 
                    doc_path="./Bioc_contribution_downloads", required_ext=[".html"], recursive=True):
    # Check if the storage exists
    PERSIST_DIR = storage_path
    if not Path(PERSIST_DIR).is_dir():
        documents = SimpleDirectoryReader(input_dir=doc_path,
                required_exts=required_ext, recursive=recursive).load_data()
        #Create an index to embed the documents and store them in the vector store
        index = VectorStoreIndex.from_documents(documents, embed_model=embed_model, 
                                                show_progress=True)
        index.storage_context.persist(persist_dir=PERSIST_DIR)
    else:
        #load the existing index
        storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
        index = load_index_from_storage(storage_context)
    return index

def query_llm(question, llm, index, judge_llm):
    Settings.llm = llm
    # Query the index
    question = question
    chat_engine_1 = index.as_chat_engine(llm=llm)

    response_1 = chat_engine_1.chat(question)
    reviewed_response = review_llm(question, response_1, judge_llm, index)
    revised_text = f"""
    The question was, {question}.
    The answer you gave was: {response_1}
    Feedback from the review: {reviewed_response}
    Task:
    Please, refine your answer by incorporating this feedback and adding more information where necessary
    """
    response_2 = chat_engine_1.chat(revised_text)
    return response_2

def query_llms(question, llms, index, judge_llm):
    llms_responses = {}
    for llm in llms:
        response = query_llm(question, llm, index, judge_llm)
        llms_responses[llm] = [response]
    return llms_responses

def review_llm(question, response, llm, index):
    chat_engine_2 = index.as_chat_engine(llm=llm)
    input_text = f"""
    The question was, {question}.
    Answer from the llm was {response}
    Supporting Documents: {response.source_nodes}
    Task: 
    Please, evaluate the correctness, relevance, and completeness of the answer and the 
    source nodes provided above. Meticulously identify any errors, suggest improvements, 
    and provide your version of the answer
    """
    response_1 = chat_engine_2.chat(input_text)
    return response_1