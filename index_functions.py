# Import all the needed packages, modules, and classes
from llama_index.llms.ollama import Ollama
from pathlib import Path
from llama_index.core import (VectorStoreIndex,
                              StorageContext,
                              SimpleDirectoryReader,
                              Settings,
                              load_index_from_storage)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def init_llms(models: list, embed_model_name: str, base_url: str = "http://localhost:11430",
               request_timeout: int = 500):
    """
    method that initializes the list of llms passed into it and returns a list 
    of initialized llms as well as the embedding model used
    
    Args:

    models (list): a list of llms to be initialized

    embed_model_name (str): the embedding model to be used to generate the vector database

    base_url (str): the local host from which ollama will run

    request_timeout (int): the time (in seconds) that ollama should be given to initialize an llm
    """
    llms = {}
    for model in models:
        # For now we are dealing with only llama and gemma. We can expand this condition
        # to accommodate more models
        if "llama" or "gemma" in model:
            llm = Ollama(base_url=base_url, model=model, request_timeout=request_timeout)
        llms[model] = llm
    embed_model = HuggingFaceEmbedding(
        model_name=embed_model_name)
    Settings.embed_model = embed_model   
    # Settings.chunk_size = 1024
    # Settings.context_window = 3800
    return llms, embed_model

def index_documents(embed_model: str, storage_path: str="./qdrant_data", 
                    doc_path: str="./Bioc_contribution_downloads", 
                    required_ext: list=[".html"], recursive: bool=True):
    """
    method that takes in a directory containing the documents to be indexed, indexes
    and stores the indexed documents, and returns the indexed vector database

    Args:

    embed_model (str): the embedding model to be used to generate the vector database

    storage_path (str): the path where the vector database will be stored

    doc_path (str): the path containing all the documents to be indexed

    required_ext (list): list containing the type of files that should be indexed 

    recursive (optional, boolean):  when set to 'True', it ensures that all the files
         in a directory are selected recursively including directories within the directory
    """
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

def query_llm(package_query_location: str, llm, index, judge_llm: str):
    """
    method that takes in an llm to reviews packages, gets feedback from a judge_llm about its review, 
    polishes its review using the response from the judge_llm and returns the polished response

    Args:

    package_query_location (str): the location of packages to be evaluated

    llm: the llm to be used for the review

    index: the vector database of our RAG store (in this case the vector database for 
        the Bioconductor guidelines)
    
    judge_llm (str): the llm that will serve as the judge. This llm will be initialized in 
        review_llm() function
    """
    Settings.llm = llm
    # get each file that we want to evaluate
    files = Path(package_query_location + "/R").glob("**/*.R")
    vignette_files = Path(package_query_location + "/vignettes").glob("**/*.Rmd")
    files.append(vignette_files)
    # make a dictionary of the responses for each file in the package
    responses = {}
    # Run each file that we want to evaluate
    for file in files:
        print (str(file) + "\n")
        # Query the file
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
        chat_engine_1 = index.as_chat_engine(llm=llm, max_iterations=100)

        response_1 = chat_engine_1.chat(question)
        print(f"first response: {response_1}")
        # use the judge llm to review the response
        reviewed_response = review_llm(question, response_1, judge_llm, index)
        print(f"judge_llm feedback: {reviewed_response}")
        revised_text = f"""
        The question was, "{question}".
        The answer you gave was "{response_1}", and the
        feedback from the review was "{reviewed_response}". 
        Please, refine your answer by incorporating this feedback and adding more information where necessary
        """
        # refine the response based on the feedback provided by the judge llm
        chat_engine_3 = index.as_chat_engine(llm=llm, max_iterations=100)
        response_2 = chat_engine_3.chat(revised_text)
        print(f"polished response: {reviewed_response}")   
        responses[str(file)] = response_2
    return responses

def query_llms(package_query_location: str, llms: list, index, judge_llm: str):
    """
    method that queries llms to review certain packages and returns a list
    of llm responses

    Args:

    package_query_location (str): the location of packages to be evaluated

    llms (dict): the dictionary of llms that will do the review

    index: the vector database of our RAG store (in this case the vector database for 
        the Bioconductor guidelines)

    judge_llm (str): the llm used as a judge to review the llm responses
    """
    llms_responses = {} 
    for llm in llms:
        print(f"llm evaluation: {llm} \n")
        responses = query_llm(package_query_location, llms[llm], index, judge_llm)
        llms_responses[llm] = responses
    return llms_responses

def review_llm(question: str, response: str, llm: str, index):
    """
    method that enables the judge llm to review the responses of the other llms and
    returns the review of the judge llm

    Args:

    question (str): the question that was passed into the llm

    response (str): the response given by the llm used for evaluation

    llm (str): the judge llm to be initialized
    
    index: the vector database of our RAG store (in this case the vector database for 
        the Bioconductor guidelines)
    """
    judge_llm = Ollama(base_url="http://localhost:11430", model=llm, request_timeout=500)
    chat_engine_2 = index.as_chat_engine(llm=judge_llm, max_iterations=100)
    print(f"judge_llm review: {llm}")
    input_text = f"""
    The question was, "{question}".
    Answer from the llm was "{response}". Is the above response correct? what else is missing from 
    the response? what can you add, if necessary? Please, meticulously provide these answers
    """
    response_1 = chat_engine_2.chat(input_text)
    return response_1