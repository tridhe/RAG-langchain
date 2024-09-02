import os
from data.data_processing import load_and_preprocess_data
from embeddings.embedding_generator import EmbeddingGenerator
from indexing.chroma_indexing import ChromaIndexing
from indexing.langchain_setup import setup_langchain_chroma
from models.llama_model import LlamaModel
from models.inference import load_llama_model, generate_response, load_saved_model, save_model
from evaluation.evaluation_metrics import evaluate_model

def main():
    # Load and preprocess the data
    dataset = load_and_preprocess_data()
    print("Dataset loaded and preprocessed:", dataset)

    # Initialize the Embedding Generator
    embedding_generator = EmbeddingGenerator()

    # Generate embeddings for context passages
    context_passages = [item['context'] for item in dataset['train']]
    context_embeddings = embedding_generator.generate_embeddings_mixed_precision(
        text_list=context_passages,
        batch_size=32
    )
    print("\nEmbeddings generated for context passages.\n")

    # Initialize Chroma indexing and add documents
    chroma_index = ChromaIndexing()
    document_ids = [item['id'] for item in dataset['train']]
    titles = [item['title'] for item in dataset['train']]
    embeddings_list = context_embeddings.tolist()

    chroma_index.add_documents(
        document_ids=document_ids,
        context_texts=context_passages,
        embeddings_list=embeddings_list,
        titles=titles
    )
    print("\nDocuments indexed in Chroma.\n")

    # Setup LangChain with Chroma
    retriever = setup_langchain_chroma(
        client=chroma_index.client,
        collection_name=chroma_index.collection_name,
        embedding_model_name="bert-base-uncased"
    )
    print("\nLangChain retriever initialized.\n")

    # Load LLaMA model for inference
    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    token = "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"
    model_save_path = "./saved_llama_model"

    # Check if the model is already saved
    if os.path.exists(model_save_path):
        print("Saved model is present")
        llama_model, llama_tokenizer = load_saved_model(model_save_path)
        print("Loaded saved model and tokenizer.")
    else:
        print("Saved model is not present")        
        llama_model, llama_tokenizer = load_llama_model(model_path, token)
        save_model(llama_model, llama_tokenizer, model_save_path)
        print("Initialized and saved model.")

    # Evaluate the model on the validation dataset
    validation_dataset = dataset['validation']
    metrics = evaluate_model(validation_dataset, retriever, llama_model, llama_tokenizer)
    print(f"Evaluation Metrics: {metrics}")


if __name__ == "__main__":
    main()
