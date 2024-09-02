from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Initialize the LLaMA 3 model and tokenizer
def save_model(model, tokenizer, save_directory):

    model.save_pretrained(save_directory)
    tokenizer.save_pretrained(save_directory)
    print(f"Model and tokenizer saved to {save_directory}")

def load_saved_model(load_directory):
    
    model = AutoModelForCausalLM.from_pretrained(load_directory)
    tokenizer = AutoTokenizer.from_pretrained(load_directory)
    return model, tokenizer

def load_llama_model(model_path: str, token: str):

    double_quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=token)
    model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=double_quant_config, token=token)
    model.eval()
    
    return model, tokenizer

def filter_unique_contexts(contexts):

    # Remove duplicate contexts
    unique_contexts = list(set(contexts))
    return unique_contexts

def generate_response(query, retriever, model, tokenizer, top_k=2):

    # Retrieve relevant documents
    retrieved_docs = retriever.get_relevant_documents(query)[:top_k]

    # Handle the case when no documents are retrieved
    if not retrieved_docs:
        return "No relevant information found for your query."

    # Extract and filter unique contexts
    contexts = [doc.page_content for doc in retrieved_docs]
    unique_contexts = filter_unique_contexts(contexts)
    combined_context = "\n\n".join(unique_contexts)

    # Prompt for the model
    input_text = f"Context:\n{combined_context}\n\nQuestion: {query}\n\nAnswer:"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=25,
            num_beams=5,
            early_stopping=True
        )

    # Decode the generated tokens
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return response

if __name__ == "__main__":
    # Example usage
    model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    token = "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"

    # Load model and tokenizer
    model, tokenizer = load_llama_model(model_path, token)

    # Assume retriever is set up elsewhere
    user_query = "In what country is Normandy located?"
    retriever = None  # Replace with actual retriever

    response = generate_response(user_query, retriever, model, tokenizer)
    print("Generated Response:", response)
