# from langchain_chroma import Chroma
# from langchain_core.prompts import ChatPromptTemplate
# from get_embedding_function import get_embedding_function
# from langchain_ollama import OllamaLLM
# from similarity_pipeline import step_1_similarity_search , step_2_similarity_search_score
# query_text = "What is chromadb?"

# PromptTemplate = """

# Answer the question onlt based on the following context:
# {context}
# -----------
 
# Answer the question based on the above context: {question}


# """

# def query_rag(query_text):
#     # prepare db
#     embedding_function=get_embedding_function()
#     db = Chroma(
#         persist_directory="chroma",
#         embedding_function=embedding_function
#     )
   
#     #sources,context = step_1_similarity_search(query_text,db)
#     sources,context = step_2_similarity_search_score(query_text,db)

#     #context = doc.page_content
    
#     # prompt tempelate
#     prompttemplate=ChatPromptTemplate.from_template(PromptTemplate)
#     prompt = prompttemplate.format(context=context, question=query_text)    
#     #print(f"prompt: {prompt}")

#     # generation model
#     model = OllamaLLM(model="mistral")
#     response_text = model.invoke(prompt)

   
#     #formating
#     formates_response = f"Answer: {response_text}\n\nSources:\n {sources}"

#     return formates_response

# print(query_rag(query_text))











# #1 context
# #2 prompt tempelate
# #3 generation model
# #4 sources
# #5 formating
# #6 testing








# print(query_rag(query_text))















from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
from get_embedding_function import get_embedding_function
from similarity_pipeline import step_2_similarity_search_score

PROMPT_TEMPLATE = """
Answer the question only based on the following context:
{context}
-----------
Answer the question based on the above context: {question}
"""

def generate_multi_queries(original_query, llm, num_queries=3):
    prompt = f"""You are an AI assistant. Your task is to generate {num_queries} 
    different versions of the given user query to retrieve relevant documents from a vector database. 
    By generating multiple perspectives on the user query, your goal is to help the user 
    overcome some of the limitations of the distance-based similarity search.
    Original query: {original_query}"""
    
    response = llm.invoke(prompt)
    # هنا بنعمل parsing للرد عشان نطلع لستة بالأسئلة
    queries = [original_query] + response.split('\n')[:num_queries]
    return queries



def query_rag(query_text: str):
    db = Chroma(
        persist_directory="chroma",
        embedding_function=get_embedding_function()
    )
   
    # Retrieval ()
    sources, context = step_2_similarity_search_score(query_text, db)

    # Prompting ()
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context, question=query_text)

    # Generation ()
    model = OllamaLLM(model="mistral")
    response_text = model.invoke(prompt)

    formatted_response = f"Answer: {response_text}\n\nSources: {sources}"
    return formatted_response

if __name__ == "__main__":
    question = "What modulation scheme is used in 4G?"  #str(input("Enter your question: "))
    print(query_rag(question))