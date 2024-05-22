"""colbert load and save module"""
__all__=["rag_colbert_save","rag_colbert_load"]
from ragatouille import RAGPretrainedModel


def rag_colbert_save(data):

    """
    it just takes langchain document(page_content,metadata) and splits the page_content based on the token size and stores splitted
    page content with metadata

    Parameters:
    -data- Document: langchain document
    
    Returns: nothing just stores the data to db

   """

    meta=data.metadata
    data=data.page_content
    RAG = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")
    RAG.index(
        collection=[data],
        index_name="cook",
        document_metadatas=[meta],
        max_document_length=300, # no of tokens 
        split_documents=True # enables the splitting
    )


def rag_colbert_load():
    
    """
    it just load already saved data from the db

    Parameters: nothing

    Returns: retriever for db
    
    """

    db=RAGPretrainedModel.from_index("./.ragatouille/colbert/indexes/cook")
    return  db.as_langchain_retriever(k=4)
