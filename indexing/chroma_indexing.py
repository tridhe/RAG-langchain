import chromadb
from chromadb.api.types import Documents, Embeddings

class ChromaIndexing:
    def __init__(self, collection_name="context_collection"):
        self.client = chromadb.Client()
        self.collection_name = collection_name
        self.collection = self.create_collection()

    def create_collection(self):
        # List all collections
        collections = self.client.list_collections()
        collection_names = [col.name for col in collections]
        
        # Delete collection if it already exists
        if self.collection_name in collection_names:
            self.client.delete_collection(self.collection_name)

        return self.client.create_collection(name=self.collection_name)

    def add_documents(self, document_ids, context_texts, embeddings_list, titles):

        self.collection.add(
            ids=document_ids,
            embeddings=embeddings_list,
            documents=context_texts,
            metadatas=[
                {"id": doc_id, "title": title}
                for doc_id, title in zip(document_ids, titles)
            ]
        )
        print(f"Added {len(document_ids)} documents to the Chroma collection.")

if __name__ == "__main__":
    from data.data_processing import load_and_preprocess_data
    from embeddings.embedding_generator import EmbeddingGenerator

    # Load and preprocess the dataset
    dataset = load_and_preprocess_data()

    # Initialize the embedding generator
    embedding_generator = EmbeddingGenerator()
    context_passages = [item['context'] for item in dataset['train']]

    # Generate embeddings
    context_embeddings = embedding_generator.generate_embeddings_mixed_precision(
        text_list=context_passages,
        batch_size=32
    )

    # Prepare metadata
    document_ids = [item['id'] for item in dataset['train']]
    titles = [item['title'] for item in dataset['train']]
    embeddings_list = context_embeddings.tolist()

    # Initialize Chroma indexing and add documents
    chroma_index = ChromaIndexing()
    chroma_index.add_documents(
        document_ids=document_ids,
        context_texts=context_passages,
        embeddings_list=embeddings_list,
        titles=titles
    )
