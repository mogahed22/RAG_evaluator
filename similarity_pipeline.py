# def step_1_similarity_search(query_text:str,db):
#     # search db
    
#      # search db
#     results = db.similarity_search(query_text,k=5)    

#     # context
#     context = "\n\n --- \n\n".join([doc.page_content for doc in results  ])
#     print(f"context: {context}")
    
#     #sources
#     sources = [doc.metadata.get("source", "unknown") for doc in results]
    

#     return sources,context


# def step_2_similarity_search_score(query_text:str,db):
#     # search db
    
#      # search db
#     results = db.similarity_search_with_score(query_text,k=5)    
#     # context

#     context = "\n\n --- \n\n".join([doc.page_content for doc,score in results  ])
#     print(f"context: {context}")
    
#     #sources
#     sources = [doc.metadata.get("source", "unknown") for doc in results]


#     return sources,context


# def step_3_similarity_search_score_with_threshold(query_text:str,db,threshold:float):
#     results = db.similarity_search_with_score(query_text,k=5)    
#     retriever_threshold = db.as_retriever(
#         search_type="similarity_score_threshold",
#         search_kwargs={"score_threshold": 0.6}
#     )

#     print(f"retriever_threshold: {retriever_threshold}")




def step_1_similarity_search(query_text: str, db, k=5):
    results = db.similarity_search(query_text, k=k)    
    context = "\n\n---\n\n".join([doc.page_content for doc in results])
    sources = [doc.metadata.get("source", "unknown") for doc in results]
    return sources, context

def step_2_similarity_search_score(query_text: str, db, k=5):
    results = db.similarity_search_with_score(query_text, k=k)    
    context = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    sources = [doc.metadata.get("source", "unknown") for doc, _score in results]
    return sources, context