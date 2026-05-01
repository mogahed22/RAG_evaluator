from langchain_ollama import OllamaEmbeddings

embedding_model_name = 'nomic-embed-text'

def get_embedding_function():
    embeddings = OllamaEmbeddings(model=embedding_model_name)
    return embeddings