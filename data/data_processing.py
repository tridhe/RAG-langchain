import json
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from datasets import Dataset, DatasetDict # type: ignore

def load_and_preprocess_data(data_path='data/squad-2.json'):
    # Load the dataset from the JSON file
    with open(data_path, 'r') as f:
        squad_2 = json.load(f)

    # Flatten the nested JSON structure
    flattened_data = []
    answers_list = []

    for article in squad_2['data']:
        title = article['title']
        for paragraph in article['paragraphs']:
            context = paragraph['context']
            for qa in paragraph['qas']:
                # Skip "impossible" questions
                if qa['is_impossible']:
                    continue

                question = qa['question']
                id_ = qa['id']
                answers = qa['answers']

                flattened_data.append({
                    'id': id_,
                    'title': title,
                    'context': context,
                    'question': question,
                    'answers': answers
                })

                for ans in answers:
                    answers_list.append(ans['text'])

    # Convert the flattened data to a DataFrame
    flattened_df = pd.DataFrame(flattened_data)

    # Split the DataFrame into training and validation sets (90-10 split)
    train_df, valid_df = train_test_split(flattened_df, test_size=0.1, random_state=42)
    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)

    # Convert the DataFrames to Hugging Face Datasets
    train_dataset = Dataset.from_pandas(train_df)
    valid_dataset = Dataset.from_pandas(valid_df)

    # Create a DatasetDict
    dataset = DatasetDict({
        'train': train_dataset,
        'validation': valid_dataset
    })

    return dataset

if __name__ == "__main__":
    dataset = load_and_preprocess_data()
    print(dataset)
