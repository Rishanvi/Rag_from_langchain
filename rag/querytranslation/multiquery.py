__all__=["rag_multiquery"]
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableBranch,RunnableLambda
from rag.models import rag_mistral_7b_ins_v2
from rag.indexing.Chromadb import rag_chroma_load
from rag.indexing.colbert import rag_colbert_load
from rag.indexing.Raptor import rag_raptor_load
from langchain.load import dumps, loads
def query_generation_chain():

    """
    it just gives query generation chain (the model gives list of questions for the user question)
    
    
    """

    template = """You are an AI language model assistant. Your task is to strictly generate ten
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines. 
    Original question: {question}
    Output:
    """

    prompt=PromptTemplate.from_template(template=template)
    return prompt | rag_mistral_7b_ins_v2() | StrOutputParser() | (lambda x: x.split("\n"))

    
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
            answers.append(dumps(item))
    answers=[loads(data) for data in set(answers)]
    return answers
def rag_multiquery(query,s='colbert'):
    """
    it takes query and db selector, generates multiple questions for the user question ,retrieves documents for multiple questions
    and filters the  retrieved documents ,returns filtered documents

    
    Parameters:
     -query- str : actual user query
     -s-   str   : db selection value

    Returns:
     -results- Document : list of filtered langchain documents 

    """
    questions=query_generation_chain().invoke({"question":query})
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