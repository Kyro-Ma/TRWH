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

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(parent_dir, 'Codes'))
from utils import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from sentence_transformers import SentenceTransformer
import pickle
import torch
import gc
import re
from tqdm import tqdm  # Import tqdm for progress bar


if __name__ == '__main__':
    beauty_user_profile_path = '../Datasets/beauty_user_profile.pkl'
    beauty_generated_user_embeddings_path = '../Datasets/beauty_generated_user_embeddings.pkl'
    fashion_user_profile_path = '../Datasets/fashion_user_profile.pkl'
    fashion_generated_user_embeddings_path = '../Datasets/fashion_generated_user_embeddings.pkl'
    user_instruction = "This is the profile of an Amazon User, detailed description of each key is as follows.\n\"summarization\": \"A summarization of what types of product this user is likely to enjoy\" (if this value is \"None\", it represents former model is unable to summarize it)\n\"reasoning\": \"briefly explain your reasoning for the summarization\""
    batch_size = 64
    device = 'cuda'
    model = SentenceTransformer("hkunlp/instructor-xl")

    user_profile_paths = [beauty_user_profile_path, fashion_user_profile_path]
    generated_user_embeddings_paths = [beauty_generated_user_embeddings_path, fashion_generated_user_embeddings_path]

    for user_profile_path, generated_user_embeddings_path in zip(user_profile_paths, generated_user_embeddings_paths):
        user_profile = load_dataset(user_profile_path)

        text_instruction_pairs = []
        for _, profile in user_profile.items():
            text = f"\"summarization\": \"{profile['summarization']}\"\n\"reasoning\": \"{profile['reasoning']}\""
            text_instruction_pairs.append({
                "instruction": user_instruction,
                "text": text
            })

        # postprocess
        texts_with_instructions = []
        for pair in text_instruction_pairs:
            texts_with_instructions.append([pair["instruction"], pair["text"]])

        generated_user_embeddings = model.encode(
            texts_with_instructions,
            batch_size=batch_size,
            show_progress_bar=True,
            output_value='sentence_embedding',
            convert_to_numpy=True,
            convert_to_tensor=False,
            device=None,
            normalize_embeddings=False
        )

        userids = list(user_profile.keys())
        user_embeddings = {}

        # the sequence of generated_user_embeddings is as same as the sequence of user_profile
        for userid, embedding in zip(userids, generated_user_embeddings):
            user_embeddings[userid] = embedding

        with open(generated_user_embeddings_path, 'wb') as f:
            pickle.dump(user_embeddings, f)