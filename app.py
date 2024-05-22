import os
from langsmith import Client
from dotenv import load_dotenv
load_dotenv()
Client()
from typing import TypedDict
from langgraph.graph import StateGraph,END
from rag.prompting import general_prompt
from rag.querytranslation import rag_multiquery,rag_decomposition,rag_stepback,rag_hyde,rag_ragfusion
from rag.indexing.Chromadb import rag_chroma_load
from rag.indexing.Raptor import rag_raptor_load
from rag.indexing.colbert import rag_colbert_load
import streamlit as st



class cook_data(TypedDict):
    query: str
    querying_tech: str
    retrieving_tech:str
    documents:list
    answer:str
    
def retrieve_documents_node(state):
    
    if(state["querying_tech"]=="multiquery"):
        state["documents"]=rag_multiquery(query=state["query"],s=state["retrieving_tech"])
    elif(state["querying_tech"]=="ragfusion"):
        state["documents"]=rag_ragfusion(query=state["query"],s=state["retrieving_tech"])
    elif(state["querying_tech"]=="decomposition"):
        state["documents"]=rag_decomposition(query=state["query"],s=state["retrieving_tech"])
    elif(state["querying_tech"]=="stepback"):
        state["documents"]=rag_stepback(query=state["query"],s=state["retrieving_tech"])
    elif(state["querying_tech"]=="hyde"):
        state["documents"]=rag_hyde(query=state["query"],s=state["retrieving_tech"])
    else:
        if(state["retrieving_tech"]=="chroma"):
            state["documents"]=rag_chroma_load().invoke(state["query"])
        elif(state["retrieving_tech"]=="raptor"):
            state["documents"]=rag_raptor_load().invoke(state["query"])
        else:
            state["documents"]=rag_colbert_load().invoke(state["query"])

    return state
def answer_generation_node(state):
    
    def format_docs(docs):
        page_format=""
        for doc in docs:
            page_format+=(doc.page_content+"\n\n")
        return page_format
    context=format_docs(state["documents"])
    print("context:",context)
    state["answer"]=general_prompt().invoke({"question":state["query"],"context":context})
    return state

graph = StateGraph(cook_data)
graph.add_node("retrieve_documents",retrieve_documents_node)
graph.add_node("answer_generation_node",answer_generation_node)
graph.set_entry_point("retrieve_documents")
graph.add_edge("retrieve_documents","answer_generation_node")
graph.add_edge("answer_generation_node",END)
app=graph.compile()



def final_answer(query,querying_tech,retrieving_tech):
    final_answer=app.invoke({"query":query,"querying_tech":querying_tech,"retrieving_tech":retrieving_tech})

    return final_answer["answer"]

side_bar=st.sidebar
query_tech= side_bar.selectbox(
   "quering technique",
   ("multiquery","ragfusion","decomposition","stepback","hyde","normal"),
   index=5
)
retrieve_tech = side_bar.selectbox(
   "retrieving technique",
   ("chroma", "raptor", "colbert"),
   index=2
)


if 'store_message' not in st.session_state:
    st.session_state.store_message=[]

st.title(":blue[Y]Ou :red[A]sk x :blue[B]ae")

container=st.container(height=400)
message=st.chat_input("say somthing.....")
demo_chat=container.chat_message(name="user",avatar="🤖")
with demo_chat:
    demo_chat.write("I' am :blue[AI] :red[BOT] # How Can I help U ?")
    
if(message):
    answer=final_answer(message,query_tech,retrieve_tech )
    st.session_state.store_message.append({"user":message,"ass":answer})
    #side_bar.write(st.session_state.store_message)    

for chat in st.session_state.store_message:
    user=container.chat_message(name="user",avatar="😉")
    with user:
        user.write(chat['user'])
    ass=container.chat_message(name="assistant",avatar="🤖")
    with ass:
        ass.write(chat['ass'])   