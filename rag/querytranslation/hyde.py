__all__=["rag_hyde"]
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.models import rag_mistral_7b_ins_v2
from rag.indexing.Chromadb import rag_chroma_load
from rag.indexing.colbert import rag_colbert_load
from rag.indexing.Raptor import rag_raptor_load
from langchain.load import dumps, loads

def query_generation_chain():
    """
    it just gives query generation chain (the model gives list of passages  for the user question)
    
    
    """

    template = """
    write a  hypothetical passage to answer the original question .

    Original question: {question}

    Output(five passages separeted hastag ):
    """

    prompt=PromptTemplate.from_template(template=template)
    return prompt | rag_mistral_7b_ins_v2() | StrOutputParser() | (lambda x: x.split("#"))


def get_docs(docs):

    """
    it takes list of langchain documents and filters the list (unique items ),
    returns filtered documents
    Parameters:
    -docs-  list : list of langchain documents 

    Returns:
    -answers- list : list of filtered langchain documents 
    
    """

    answers=[]
    for doc in docs:
        for item in doc:
            print(item)
            answers.append(dumps(item))
    print(len(answers))
    answers=[loads(data) for data in set(answers)]
    print(len(answers))
    return answers
def rag_hyde(query,s='3'):

    """
    it takes query and db selector, generates multiple passages for the user question ,retrieve documents for multiple passages
    and filter the  retrieved documents ,returns filtered documents

    
    Parameters:
     -query- str : actual user query
     -s-   str   : db selection value

    Returns:
     -results- Document : list of filtered langchain documents 

    """

    questions=query_generation_chain().invoke({"question":query})[1:]
    questions.insert(0,query)
    print(questions)
    if(s=="chroma"):
        chain=rag_chroma_load().map()
    elif(s=="raptor"):
        chain=rag_raptor_load().map()
    else:
        chain=rag_colbert_load().map()
    docs=chain.invoke(questions)
    results=get_docs(docs)
    return results
