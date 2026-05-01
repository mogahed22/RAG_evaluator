import os
import shutil
import argparse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from get_embedding_function import get_embedding_function

DATA_PATH = "data/golden_test_set_squad.csv"
CHROMA_PATH = "chroma"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_size", type=int, default=100)
    parser.add_argument("--chunk_overlap", type=int, default=20)
    args = parser.parse_args()
    if os.path.exists(CHROMA_PATH):
        print(f"Cleaning up old database at {CHROMA_PATH}...")
        shutil.rmtree(CHROMA_PATH)

    add_to_chroma(chunks)
    print(f"Creating database with chunk_size: {args.chunk_size} and overlap: {args.chunk_overlap}")
    documents = load_documents()
    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)

def load_documents():
    loader = CSVLoader(
        DATA_PATH,
         source_column="expected_source",
           encoding="utf-8")
    return loader.load()

def split_documents(documents: list[Document], chunk_size: int, chunk_overlap: int):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)
def add_to_chroma(chunks):
    db = Chroma(
        persist_directory=CHROMA_PATH, 
        embedding_function=get_embedding_function()
    )
    print(f"Adding {len(chunks)} chunks to {CHROMA_PATH}...")
    db.add_documents(chunks)
    print("Done!")

if __name__ == "__main__":
    main()