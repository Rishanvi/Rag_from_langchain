__all__=["rag_decomposition"]
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.models import rag_mistral_7b_ins_v2
from rag.indexing.Chromadb import rag_chroma_load
from rag.indexing.colbert import rag_colbert_load
from rag.indexing.Raptor import rag_raptor_load
from langchain.schema.document import Document

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

def get_docs(questions,docs):

    """
    it takes set of questions and documents ,performs generation with questions and documents,
    returns set of generated answers

    Parameters:
    -questions- list: list of questions
    -docs- list : list of documents

    Returns:
      list : set of generated answers
    
    """



    answer_template=""" 

    since you are a chat bot,take the given context  as a reference and give the answer
    to the user question 
    
    user_Question: {question}
    
    contex:{context}
    
    """
    answer_prompt=PromptTemplate.from_template(template=answer_template)
    answer_chain= answer_prompt| rag_mistral_7b_ins_v2() | StrOutputParser()
    answer=[]


    def format_docs(docs):
        formatted_doc=""
        for doc in docs:
            formatted_doc+=doc.page_content
        return formatted_doc

    for que,doc in zip(questions,docs):
        con=format_docs(doc)
        answer.append(answer_chain.invoke({"question":que,"context":con}))
        print(con)
    return [Document(page_content=ans) for ans in answer]

def rag_decomposition(query,s='3'):
    """
    it takes query and db selector, generates multiple questions for the user question ,retrieves documents for multiple questions
    ,generates answers based on the multiple questions and retrieved documents ,returns set of answers

    
    Parameters:
     -query- str : actual user query
     -s-   str   : db selection value

    Returns:
     -results- Document : list of langchain documents (with a generated answers )

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
    results=get_docs(questions,docs)
    return results
    