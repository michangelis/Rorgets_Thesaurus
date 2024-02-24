from langchain_openai import OpenAIEmbeddings
import json
import os

def load_existing_data(filename):
    if os.path.exists(filename):
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def process_and_save_embeddings(original_data, output_filename):

    OPEN_AI_API_KEY = "YOUR_OPENAI_API_KEY"

    # Initialize the OpenAIEmbeddings with your API key and desired model
    embeddings_model = OpenAIEmbeddings(openai_api_key=OPEN_AI_API_KEY, model="text-embedding-ada-002")

    # Load any existing embeddings data
    new_structure = load_existing_data(output_filename)
    
    for class_name, sections in original_data.items():
        if class_name not in new_structure:
            new_structure[class_name] = {}

        for section_name, words in sections.items():
            # Skip sections that have already been processed
            if section_name in new_structure[class_name]:
                print(f"Skipping {class_name} - {section_name}, already processed.")
                continue

            print(f"Processing {class_name} - {section_name}")
            new_structure[class_name][section_name] = []
            
            for word_group in words:
                # Try-except block to handle potential errors during embedding
                try:
                    embeddings = embeddings_model.embed_documents(word_group)
                    new_structure[class_name][section_name].append(embeddings)
                except Exception as e:
                    print(f"Error processing {class_name} - {section_name}: {e}")
                    # Optionally, break or continue based on your error handling strategy

            # Save the updated structure after processing each section
            with open(output_filename, 'w', encoding='utf-8') as f:
                json.dump(new_structure, f, ensure_ascii=False, indent=4)


