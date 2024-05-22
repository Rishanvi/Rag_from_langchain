"""models selection module"""
__all__=["rag_openai","rag_chatopenai","rag_mistral_7b_ins_v2","rag_mistral_7b_v1","rag_mistral_8x7b_ins_v1","rag_all_mini_l6_v2","rag_all_mini_l12_v2"]
from langchain_openai import OpenAI,ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain_community.embeddings import HuggingFaceEmbeddings


rag_openai=lambda : OpenAI()
rag_chatopenai=lambda : ChatOpenAI()
rag_mistral_7b_ins_v2=lambda: HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-Instruct-v0.2",max_length=200,temperature=0.2)
rag_mistral_7b_v1=lambda : HuggingFaceEndpoint(repo_id="mistralai/Mistral-7B-v0.1",max_length=200,temperature=0.2)
rag_mistral_8x7b_ins_v1= lambda :HuggingFaceEndpoint(repo_id="mistralai/Mixtral-8x7B-Instruct-v0.1",max_length=200,temperature=0.2)
rag_all_mini_l6_v2=lambda : HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
rag_all_mini_l12_v2=lambda : HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
