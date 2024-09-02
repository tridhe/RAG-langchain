from transformers import BertTokenizer, BertModel
import gc
import torch
import numpy as np
from tqdm import tqdm
import os

class EmbeddingGenerator:
    def __init__(self, model_name='bert-base-uncased', embeddings_dir='embeddings'):

        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to('cuda')
        self.embeddings_dir = embeddings_dir
        os.makedirs(embeddings_dir, exist_ok=True)

    def _get_embeddings_file_path(self, name):
        return os.path.join(self.embeddings_dir, f"{name}.npy")

    def generate_embeddings_mixed_precision(self, text_list, batch_size=32, max_length=512, save_as='context_embeddings'):

        embeddings_file_path = self._get_embeddings_file_path(save_as)
        # Check if embeddings already exist
        print('here')
        if os.path.exists(embeddings_file_path):
            print(f"Loading embeddings from {embeddings_file_path}")
            return np.load(embeddings_file_path)

        all_embeddings = []
        self.model.eval()

        for i in tqdm(range(0, len(text_list), batch_size)):
            batch_texts = text_list[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=max_length).to('cuda')

            with torch.no_grad():
                with torch.cuda.amp.autocast():  # Enable mixed precision
                    outputs = self.model(**inputs)

            batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeddings.append(batch_embeddings)

            # Clear cache and free memory
            del inputs, outputs, batch_embeddings
            torch.cuda.empty_cache()
            gc.collect()

        all_embeddings = np.vstack(all_embeddings)

        # Save embeddings to file
        np.save(embeddings_file_path, all_embeddings)
        print(f"Saving embeddings to {embeddings_file_path}")

        return all_embeddings