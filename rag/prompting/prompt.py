"""prompting module"""

__all__=["general_prompt"]

from langchain.prompts import PromptTemplate
from rag.models import rag_mistral_7b_ins_v2,rag_openai
from langchain_core.output_parsers import StrOutputParser

def general_prompt():
    """
    it just returns general prompt->model->outputparser chain for answer generation 
    
    """
    
    template="""
        Considering the context provided, generate a well-structured and informative response that addresses the question comprehensively. 
        Ensure that your response is concise yet sufficiently detailed, providing relevant insights and information pertinent to the given context.
        Strive for clarity and coherence in your answer, maintaining a balance between brevity and depth to convey the key points effectively.

        Context:{context} 

        Question:{question}

        """

    prompt=PromptTemplate.from_template(template=template)


    return prompt | rag_mistral_7b_ins_v2() | StrOutputParser()

