from typing import List
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from langchain.vectorstores import Chroma as LangChainChroma
import chromadb  # Corrected import for ChromaDB

# Initialize the TransformerEmbeddings class with a model and tokenizer.
class TransformerEmbeddings:
    def __init__(self, model_name: str, max_length=1024):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        self.model.eval()

    def embed_query(self, query: str) -> np.ndarray:

        return self.embed_documents([query])[0]

    def embed_documents(self, documents: List[str]) -> List[np.ndarray]:

        inputs = self.tokenizer(documents, padding=True, truncation=True, return_tensors="pt", max_length=self.max_length)

        with torch.no_grad():
            outputs = self.model(**inputs)

        embeddings = outputs.last_hidden_state[:, 0, :].numpy()

        return embeddings.tolist()

# Function to set up LangChain with Chroma
def setup_langchain_chroma(client, collection_name: str, embedding_model_name: str):

    # Initialize the embedding function
    embedding_function = TransformerEmbeddings(model_name=embedding_model_name)

    # Initialize LangChain's Chroma wrapper
    langchain_chroma = LangChainChroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embedding_function
    )

    # Initialize the retriever
    retriever = langchain_chroma.as_retriever()

    return retriever

if __name__ == "__main__":
    from chroma_indexing import ChromaIndexing

    # Example usage
    chroma_client = chromadb.Client()  # Corrected initialization of Chroma client
    collection_name = "context_collection"
    embedding_model_name = "bert-base-uncased"

    retriever = setup_langchain_chroma(chroma_client, collection_name, embedding_model_name)
    print("LangChain setup with Chroma complete.")
