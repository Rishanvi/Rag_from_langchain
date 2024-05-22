"""document loader module"""
__all__=["rag_webbase_loader","rag_pdf_loader"]

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader

def rag_webbase_loader(url):
    """
    scrape the webpage from given url 

    Parameters:
    -url- str : website url

    Returns:
    -data- Document : webscraped data with page_content  and metadata
    """
    loader = WebBaseLoader(url)
    data = loader.load()
    return data

def rag_pdf_loader(path):
    """
    scrape the tetx data  from given path(.pdf file) 

    Parameters:
    -path- str : pdf file path

    Returns:
    -data- Document : pdf scrapped text data (page_content) and metadata
    """
    loader = PyPDFLoader(path)
    data = loader.load()
    return data
