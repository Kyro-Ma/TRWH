import gc
import sys
import json
import ast

'''
In local, run import below
'''
# from Codes.utils import load_dataset
'''
In linux, run import below
'''
import os
import pandas as pd
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'Codes'))
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from sentence_transformers import SentenceTransformer
import pickle
import torch
import gc
import re
from tqdm import tqdm  # Import tqdm for progress bar
import torch.nn as nn

def load_dataset(file_path):
    df = pd.read_pickle(file_path)
    return df

if __name__ == '__main__':
    beauty_item_profile_path = '../Datasets/beauty_item_profile.pkl'
    beauty_generated_item_embeddings_path = '../Datasets/beauty_generated_item_embeddings.pkl'
    fashion_item_profile_path = '../Datasets/fashion_item_profile.pkl'
    fashion_generated_item_embeddings_path = '../Datasets/fashion_generated_item_embeddings.pkl'
    item_instruction = "This is the profile of an Amazon Product, detailed description of each key is as follows.\n\"summarization\": \"A summarization of what types of users would enjoy this product\" (if this value is \"None\", it represents former model is unable to summarize it)\n\"reasoning\": \"briefly explain your reasoning for the summarization\""
    batch_size = 64
    device = torch.device("cuda:7")
    model = SentenceTransformer("hkunlp/instructor-xl")

    item_profile_paths = [beauty_item_profile_path, fashion_item_profile_path]
    generated_item_embeddings_paths = [beauty_generated_item_embeddings_path, fashion_generated_item_embeddings_path]

    for item_profile_path, generated_item_embeddings_path in zip(item_profile_paths, generated_item_embeddings_paths):
        item_profile = load_dataset(item_profile_path)

        text_instruction_pairs = []
        for _, profile in item_profile.items():
            text = f"\"summarization\": \"{profile['summarization']}\"\n\"reasoning\": \"{profile['reasoning']}\""
            text_instruction_pairs.append({
                "instruction": item_instruction,
                "text": text
            })

        # postprocess
        texts_with_instructions = []
        for pair in text_instruction_pairs:
            texts_with_instructions.append([pair["instruction"], pair["text"]])

        # calculate embeddings
        generated_item_embeddings = model.encode(
            texts_with_instructions,
            batch_size=batch_size,
            show_progress_bar=True,
            output_value='sentence_embedding',
            convert_to_numpy=True,
            convert_to_tensor=False,
            device=device,
            normalize_embeddings=False
        )

        itemids = list(item_profile.keys())
        item_embeddings = {}

        # the sequence of generated_item_embeddings is as same as the sequence of item_profile
        for itemid, embedding in zip(itemids, generated_item_embeddings):
            item_embeddings[itemid] = embedding

        with open(generated_item_embeddings_path, 'wb') as f:
            pickle.dump(item_embeddings, f)