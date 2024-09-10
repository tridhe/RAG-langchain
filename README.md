
# Retrieval-Augmented Generation (RAG) System with LangChain and Chroma

## Project Overview

This project implements a Retrieval-Augmented Generation (RAG) system that retrieves information from a set of news articles across various categories and generates coherent and relevant responses to user queries. The system leverages LangChain for language modeling and Chroma for vector storage and retrieval.

### Key Features

- **Data Preprocessing**: Loads and preprocesses the SQuAD 2.0 dataset for training and validation.
- **Embeddings Generation**: Uses a pre-trained BERT model to generate embeddings for context passages.
- **Document Indexing**: Indexes the documents using Chroma vector database for efficient retrieval.
- **Query Handling and Response Generation**: Utilizes a LLaMA3 language model to generate responses based on retrieved documents.
- **Model Evaluation**: Evaluates the system's performance using standard NLP metrics.

## Project Structure

The project is organized into multiple Python files, each responsible for a specific part of the system:

```
.
├── data/
│   └── data_processing.py       # Dataset loading and preprocessing
│   └── squad-2.json             # Squad 2.0 Dataset 
├── embeddings/
│   └── embedding_generator.py   # Embedding generation using BERT
├── indexing/
│   ├── chroma_indexing.py       # Document indexing using Chroma
│   └── langchain_setup.py       # LangChain setup with Chroma
├── models/
│   └── inference.py             # Inference using LLaMA3 model
├── evaluation/
│   └── evaluation_metrics.py    # Evaluation metrics of the model
├── main.py                      # Main script to run the RAG system
├── evaluate.py                  # Script to evaluate the RAG system
├── requirements.txt             # Python dependencies for the project
└── README.md                    # Project documentation
```

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/tridhe/RAG-langchain.git
cd rag-system
```

### 2. Install Python Dependencies

Ensure you have Python 3.7+ installed. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

### 3. Run the RAG System

To run the RAG system and generate a response to a query, use the following command:

```bash
python main.py "In what country is Normandy located?"
```

Or, if you are using a Jupyter Notebook, you can use:

```python
%run main.py "In what country is Normandy located?"
```
The first run will take more time compared to the subsequent runs.


### 4. Evaluate the RAG System

To evaluate the RAG system, use the following command:

```bash
python evaluate.py
```

Or, if you are using a Jupyter Notebook, you can use:

```python
%run evaluate.py
```

## Detailed Steps

### Data Loading and Preprocessing

The dataset is loaded from a JSON file and preprocessed to extract relevant information such as the `id`, `title`, `context`, `question`, and `answers`. The preprocessed data is split into training and validation sets and converted into Hugging Face Datasets for further processing.

### Embedding Generation

Using a pre-trained BERT model (`bert-base-uncased`), embeddings are generated for the context passages. The embeddings are computed in batches and stored for indexing.

### Document Indexing with Chroma

The context embeddings and associated metadata (such as document IDs and titles) are indexed using Chroma, a vector database optimized for fast and efficient retrieval.

### Query Handling and Response Generation

A LLaMA3 model is used to generate responses to user queries based on the retrieved documents. The model is fine-tuned for question-answering tasks and integrates with the retrieval system to provide coherent responses.

### Model Evaluation

The system's performance is evaluated using BLEU and ROUGE as evaluation metrics for the generated LLM outputs. The BLEU score has been adjusted by using a custom weight distribution to emphasize unigram (0.5) and bigram (0.3) matches while reducing the weight of higher-order n-grams (3-gram: 0.15, 4-gram: 0.05). This approach better reflects the model's ability to generate unique, paraphrased answers while maintaining the essential meaning. The ROUGE score is used to complement BLEU, providing additional insight into the quality of the model's responses
Evaluation Results:
    - BLEU: 1.18
    - Average ROUGE-L: 2.37


## Requirements

The `requirements.txt` file includes all necessary Python libraries for the project:

```plaintext
torch
transformers
langchain
chromadb
datasets
transformers==4.44.2 
bitsandbytes
accelerate
langchain-community
scikit-learn
numpy
pandas
tqdm
```

Make sure to install these dependencies using the command provided in the Setup Instructions.

## Notes

- Ensure your system has adequate memory and computational resources, especially when using large language models like LLaMA3.
- The code is designed to run on a GPU for faster processing, particularly for embedding generation and response generation tasks.

## Acknowledgments

- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [Chroma](https://docs.trychroma.com/)
- [LangChain](https://docs.langchain.com/)
