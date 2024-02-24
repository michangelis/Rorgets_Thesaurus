import json
import torch
from transformers import BertModel, BertTokenizer


# Download pre-trained BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the data from the JSON file
with open('roget_words.json', 'r') as f:
    data = json.load(f)

# Dictionary to store embeddings
embeddings_dict = {}

# Process each class and section
for class_name, sections in data.items():
    class_embeddings = {}
    max_sequence_length = 0  # Track the maximum sequence length
    for section_name, word_lists in sections.items():
        section_embeddings = []
        for word_list in word_lists:
            # Tokenize words
            tokens = tokenizer(word_list, padding=True, truncation=True, return_tensors='pt')
            # Get BERT embeddings
            with torch.no_grad():
                outputs = model(**tokens)
                embeddings = outputs.last_hidden_state.mean(dim=1)  # Aggregate embeddings (mean pooling)
            section_embeddings.append(embeddings.tolist())  # Convert embeddings to list and append
        # Update max_sequence_length
        max_sequence_length = max(max_sequence_length, len(section_embeddings))
        # Store section embeddings in the dictionary
        class_embeddings[section_name] = section_embeddings
    # Pad or truncate section_embeddings to max_sequence_length
    for section_name, embeddings_list in class_embeddings.items():
        while len(embeddings_list) < max_sequence_length:
            embeddings_list.append([0] * len(embeddings_list[0]))  # Pad with zeros
        class_embeddings[section_name] = embeddings_list[:max_sequence_length]  # Truncate if necessary
    # Store class embeddings in the dictionary
    embeddings_dict[class_name] = class_embeddings

print("Data processed successfully!")

# Save embeddings to a JSON file
with open('embeddings.json', 'w') as f:
    json.dump(embeddings_dict, f)

print("Embeddings saved successfully!")

