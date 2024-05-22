""" text and document splitter module"""
__all__=["rag_recursive_text_splitter","rag_recursive_doc_splitter","rag_document_formatter"]
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document

def rag_recursive_text_splitter(data):
    """
    split the text into segments or list of text

    Parameters:
    -data- str: raw text

    Returns:
    -data- list : list of splitted text data
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )
    data=text_splitter.split_text(data)
    return data
def rag_recursive_doc_splitter(data):
    """
    split the document into segments or list of langchain documents

    Parameters:
    -data- Document: langchain doument

    Returns:
    -data- list : list of splitted langchain documents
    """

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=100,
    )
    data=text_splitter.split_documents(data)
    return data

def rag_document_formatter(data):
    """
    removing the unwanted new lines and returns formetted text(page_content) with metadata as langchain document

    Parameters:
    -data- Document: langchain document

    Returns:
    -data- Document: langchain document with newline removed page_content and metadata
    """

    meta=data[0].metadata
    data="".join(data[0].page_content.split("\n"))
    data=[Document(page_content=data,metadata=meta)]
    return data