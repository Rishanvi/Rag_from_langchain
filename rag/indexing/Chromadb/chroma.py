"""chromadb save and load module"""
__all__=["rag_chroma_load","rag_chroma_save"]
import chromadb
from rag.models import rag_all_mini_l6_v2
from rag.docloaders import rag_webbase_loader
from rag.docsplitter import rag_recursive_text_splitter
from langchain_community.vectorstores import Chroma
from langchain.schema.document import Document

def rag_chroma_save(data):
   """
   it just takes langchain document(page_content,metadata) and splits the page_content based on the chunk size and stores splitted
   page content with metadata

   Parameters:
   -data- Document: langchain document
    
   Returns: nothing just stores the data to db

   """
   meta=data.metadata
   docs=data.page_content
   docs=rag_recursive_text_splitter(docs)
   def meta_return():
        return meta
   docs=[Document(page_content=tex,metadata=meta_return()) for tex in docs]
   db2 = Chroma.from_documents(docs, rag_all_mini_l6_v2(),collection_name="demo", persist_directory="./cook")

def rag_chroma_load():
    """
    it just load already saved data from the db

    Parameters: nothing

    Returns: retriever for db
    
    """

    db3 = Chroma(persist_directory="./cook", embedding_function=rag_all_mini_l6_v2(),collection_name="demo")
    return db3.as_retriever(search_kwargs={"k":4})


