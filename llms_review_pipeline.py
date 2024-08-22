from index_functions import *

models = ["llama3.1:latest", "llama3", "gemma:2b", "gemma:7b", "Mixtral-8x22B-v0.1", "bloom", "falcon", "mixtral", "bert"]
embed_model_name = "BAAI/bge-small-en-v1.5"

llms, embed_model = init_llms(models, embed_model_name)

index = index_documents(embed_model)

question = ""
judge_llm = ""
llms_responses = query_llms(question, llms, index, judge_llm)
for llm, response in llms_responses:
    print(llm + "\n" + response)