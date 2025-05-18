import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import gc
import sys
from collections import defaultdict
from utils import load_dataset
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import warnings
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.data import HeteroData
import torch
import torch.nn.functional as F
import torch.optim as optim
import pickle
import random
import nltk
import argparse
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch.distributed as dist
from torch.multiprocessing import spawn, Manager
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import datetime


# warnings.filterwarnings("ignore", category=FutureWarning, message="'DataFrame.swapaxes' is deprecated")


def worker(rank, world_size, texts_with_instructions, return_list):
    try:
        # os.environ['MASTER_ADDR'] = 'localhost'
        # os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(rank)
        timeout = datetime.timedelta(seconds=360000)

        dist.init_process_group(
            backend='nccl',
            init_method="tcp://127.0.0.1:23456",
            world_size=world_size,
            rank=rank,
            timeout=timeout
        )

        texts_with_instructions = texts_with_instructions[rank::world_size]
        sentence_model = SentenceTransformer("hkunlp/instructor-xl").to(rank)
        sentence_model = DDP(sentence_model, device_ids=[rank])

        generated_text_embeddings = sentence_model.module.encode(
            texts_with_instructions,
            batch_size=64,
            show_progress_bar=True,
            output_value='sentence_embedding',
            convert_to_numpy=True,
            convert_to_tensor=False,
            device=rank,
            normalize_embeddings=False
        )

        local_review_train = []
        for embedding in generated_text_embeddings:
            local_review_train.append(embedding)

        gathered = [None for _ in range(world_size)]
        dist.all_gather_object(gathered, local_review_train)

        if dist.get_rank() == 0:
            review_train = []
            # each local list is in the order of your stride‐slice,
            # so you interleave by index:
            max_len = max(len(lst) for lst in gathered)
            for i in range(max_len):
                for r in range(world_size):
                    if i < len(gathered[r]):
                        review_train.append(gathered[r][i])
            return_list.extend(review_train)
    finally:
        dist.destroy_process_group()


def train_and_evaluate(training_data, testing_data, items_dict, user_embeddings, item_embeddings):
    data_train = HeteroData()
    data_test = HeteroData()
    print_counter = 20000

    uid_train = {}
    iid_train = {}
    current_uid_train = 0
    current_iid_train = 0
    rate_count_train = len(training_data)
    counter = 0

    # map the id of user and items to numerical value
    for index, row in training_data.iterrows():
        # if counter % print_counter == 0:
        #     print(str(round(counter / rate_count_train, 2) * 100) + '%')

        if row['user_id'] in uid_train.keys():
            pass
        else:
            uid_train[row['user_id']] = current_uid_train
            current_uid_train += 1

        if row['parent_asin'] in iid_train:
            pass
        else:
            iid_train[row['parent_asin']] = current_iid_train
            current_iid_train += 1

        counter += 1

    uid_test = {}
    iid_test = {}
    current_uid_test = 0
    current_iid_test = 0
    rate_count_test = len(testing_data)
    counter = 0
    # print("standardise user id and item id for testing train_edge_data")
    for index, row in testing_data.iterrows():
        # if counter % print_counter == 0:
        #     print(str(round(counter / rate_count_test, 2) * 100) + '%')

        if row['user_id'] in uid_test.keys():
            pass
        else:
            uid_test[row['user_id']] = current_uid_test
            current_uid_test += 1

        if row['parent_asin'] in iid_test:
            pass
        else:
            iid_test[row['parent_asin']] = current_iid_test
            current_iid_test += 1

        counter += 1

    # Add user node IDs (without features)
    data_train['user'].num_nodes = current_uid_train  # Number of users
    data_test['user'].num_nodes = current_uid_test
    item_features_train = []
    item_features_test = []
    counter = 0
    # print("Getting item features (training)")
    for value in iid_train.keys():
        # if counter % print_counter == 0:
        #     print(str(round(counter / len(iid_train.keys()), 2) * 100) + '%')

        target = items_dict[value]
        # temp = [target['average_rating'], target['rating_number']] + target['embed'].tolist()
        temp = target['embed'].tolist()
        item_features_train.append(temp)
        counter += 1

    counter = 0
    # print("Getting item features (testing)")
    for value in iid_test.keys():
        # if counter % print_counter == 0:
        #     print(str(round(counter / len(iid_test.keys()), 2) * 100) + '%')

        target = items_dict[value]
        # temp = [target['average_rating'], target['rating_number']] + target['embed'].tolist()
        temp = target['embed'].tolist()
        item_features_test.append(temp)
        counter += 1

    # Adding item nodes with features
    data_train['item'].x = torch.tensor(item_features_train, dtype=torch.float).to(device)  # Item features (2D)
    data_test['item'].x = torch.tensor(item_features_test, dtype=torch.float).to(device)  # Item features (2D)

    # region training edges
    rating_edge_from_train, rating_edge_to_train = [], []
    rating_train = []
    verify_buy_from_train, verify_buy_to_train = [], []
    review_list_train = []
    review_train = []
    review_edge_from_train, review_edge_to_train = [], []
    counter = 0
    store_item_dict_train = {key: [] for key in stores}
    same_store_edge_train = [[], []]

    for index, row in training_data.iterrows():
        # if counter % print_counter == 0:
        #     print(str(round(counter / len(iid_test.keys()), 2) * 100) + '%')

        rating_edge_from_train.append(uid_train[row['user_id']])
        rating_edge_to_train.append(iid_train[row['parent_asin']])
        rating_train.append(row['rating'])
        store_item_dict_train[items_dict[row['parent_asin']]['store']].append(iid_train[row['parent_asin']])

        if row['text'] is not None:
            review_edge_from_train.append(uid_train[row['user_id']])
            review_edge_to_train.append(iid_train[row['parent_asin']])
            review_list_train.append('Title:' + row['title'] + 'Text:' + row['text'])
            # review_train.append(get_word2vec_sentence_vector(row['title'] + row['text'], w2v_model, vector_size))

        if row['verified_purchase']:
            verify_buy_from_train.append(uid_train[row['user_id']])
            verify_buy_to_train.append(iid_train[row['parent_asin']])

        counter += 1

    texts_with_instructions = []
    for review in review_list_train:
        texts_with_instructions.append([review_instruction, review])

    return_list = manager.list()
    mp.set_start_method('fork', force=True)
    spawn(worker, args=(world_size, texts_with_instructions, return_list), nprocs=world_size, join=True)
    review_train = list(return_list)

    # solve the repeated items in the store_item_dict and build same store edge
    for store in store_item_dict_train.keys():
        item_from_store = list(set(store_item_dict_train[store]))

        if len(item_from_store) < 2:
            pass
        for i in range(len(item_from_store)):
            for j in range(i, len(item_from_store)):
                same_store_edge_train[0].append(item_from_store[i])
                same_store_edge_train[1].append(item_from_store[j])

    # Convert List of NumPy Arrays to a Single NumPy Array
    review_train = np.array(review_train).tolist()

    # Adding edges and edge attributes
    data_train['user', 'rates', 'item'].edge_index = torch.tensor(
        [rating_edge_from_train, rating_edge_to_train], dtype=torch.long
    ).to(device)
    data_train['user', 'rates', 'item'].edge_attr = torch.tensor(rating_train, dtype=torch.float).to(device)
    data_train['item', 'rated_by', 'user'].edge_index = torch.tensor(
        [rating_edge_to_train, rating_edge_from_train], dtype=torch.long
    ).to(device)
    rating_train.reverse()
    data_train['item', 'rated_by', 'user'].edge_attr = torch.tensor(
        rating_train, dtype=torch.float
    ).to(device)

    data_train['user', 'review', 'item'].edge_index = torch.tensor(
        [review_edge_from_train, review_edge_to_train], dtype=torch.long
    ).to(device)
    data_train['user', 'review', 'item'].edge_attr = torch.tensor(review_train, dtype=torch.float).to(device)
    data_train['item', 'reviewed_by', 'user'].edge_index = torch.tensor(
        [review_edge_to_train, review_edge_from_train], dtype=torch.long
    ).to(device)
    review_train.reverse()
    data_train['item', 'reviewed_by', 'user'].edge_attr = torch.tensor(review_train, dtype=torch.float).to(device)

    data_train['user', 'buys', 'item'].edge_index = torch.tensor(
        [verify_buy_from_train, verify_buy_to_train]
    ).to(device)
    data_train['item', 'bought_by', 'user'].edge_index = torch.tensor(
        [verify_buy_to_train, verify_buy_from_train]
    ).to(device)
    # item_random_walk_train = random_walk(data_train['item', 'rated_by', 'user']['edge_index'])
    # user_random_walk_train = random_walk(data_train['user', 'rates', 'item']['edge_index'])
    # data_train['user', 'related_to', 'user'].edge_index = torch.tensor(
    #     [user_random_walk_train[0] + user_random_walk_train[1],
    #      user_random_walk_train[1] + user_random_walk_train[0]]).to(device)
    # data_train['item', 'related_to', 'item'].edge_index = torch.tensor(
    #     [item_random_walk_train[0] + item_random_walk_train[1],
    #      item_random_walk_train[1] + item_random_walk_train[0]]).to(device)
    # build bidirectional edges for items within same store
    data_train['item', 'same_store', 'item'].edge_index = torch.tensor(
        [same_store_edge_train[0] + same_store_edge_train[1], same_store_edge_train[1] + same_store_edge_train[0]]
    ).to(device)

    # print('train edge data finished')

    # region testing edges
    review_list_test = []
    rating_edge_from_test, rating_edge_to_test = [], []
    rating_test = []
    verify_buy_from_test, verify_buy_to_test = [], []
    review_test = []
    review_edge_from_test, review_edge_to_test = [], []
    counter = 0
    store_item_dict_test = {key: [] for key in stores}
    same_store_edge_test = [[], []]
    for index, row in testing_data.iterrows():
        # if counter % print_counter == 0:
        #     print(str(round(counter / rate_count_test, 2) * 100) + '%')

        rating_edge_from_test.append(uid_test[row['user_id']])
        rating_edge_to_test.append(iid_test[row['parent_asin']])
        rating_test.append(row['rating'])
        store_item_dict_test[items_dict[row['parent_asin']]['store']].append(iid_test[row['parent_asin']])

        if row['text'] is not None:
            review_edge_from_test.append(uid_test[row['user_id']])
            review_edge_to_test.append(iid_test[row['parent_asin']])
            review_list_test.append('Title: ' + row['title'] + 'Text: ' + row['text'])
            # review_test.append(get_word2vec_sentence_vector(row['title'] + row['text'], w2v_model, vector_size))

        if row['verified_purchase']:
            verify_buy_from_test.append(uid_test[row['user_id']])
            verify_buy_to_test.append(iid_test[row['parent_asin']])

        counter += 1

    texts_with_instructions = []
    for review in review_list_test:
        texts_with_instructions.append([review_instruction, review])

    return_list = manager.list()
    mp.set_start_method('fork', force=True)
    spawn(worker, args=(world_size, texts_with_instructions, return_list), nprocs=world_size, join=True)
    review_test = list(return_list)

    for store in store_item_dict_test.keys():
        item_from_store = list(set(store_item_dict_test[store]))
        if len(item_from_store) < 2:
            pass
        for i in range(len(item_from_store)):
            for j in range(i, len(item_from_store)):
                same_store_edge_test[0].append(item_from_store[i])
                same_store_edge_test[1].append(item_from_store[j])

    # Convert List of NumPy Arrays to a Single NumPy Array
    review_test = np.array(review_test).tolist()

    # Adding edges and edge attributes
    data_test['user', 'rates', 'item'].edge_index = torch.tensor(
        [rating_edge_from_test, rating_edge_to_test], dtype=torch.long
    ).to(device)
    data_test['user', 'rates', 'item'].edge_attr = torch.tensor(rating_test, dtype=torch.float).to(device)
    data_test['item', 'rated_by', 'user'].edge_index = torch.tensor(
        [rating_edge_to_test, rating_edge_from_test], dtype=torch.long
    ).to(device)
    rating_test.reverse()
    data_test['item', 'rated_by', 'user'].edge_attr = torch.tensor(
        rating_test, dtype=torch.float
    ).to(device)

    data_test['user', 'review', 'item'].edge_index = torch.tensor(
        [review_edge_from_test, review_edge_to_test], dtype=torch.long
    ).to(device).to(torch.int64)
    data_test['user', 'review', 'item'].edge_attr = torch.tensor(review_test, dtype=torch.float).to(device)
    data_test['item', 'reviewed_by', 'user'].edge_index = torch.tensor(
        [review_edge_to_test, review_edge_from_test], dtype=torch.long
    ).to(device).to(torch.int64)
    review_test.reverse()
    data_test['item', 'reviewed_by', 'user'].edge_attr = torch.tensor(review_test, dtype=torch.float).to(device)

    data_test['user', 'buys', 'item'].edge_index = torch.tensor(
        [verify_buy_from_test, verify_buy_to_test]
    ).to(device).to(torch.int64)
    data_test['item', 'bought_by', 'user'].edge_index = torch.tensor(
        [verify_buy_to_test, verify_buy_from_test]
    ).to(device).to(torch.int64)
    # item_random_walk_test = random_walk(data_test['item', 'rated_by', 'user']['edge_index'])
    # user_random_walk_test = random_walk(data_test['user', 'rates', 'item']['edge_index'])
    # data_test['user', 'related_to', 'user'].edge_index = torch.tensor(
    #     [user_random_walk_test[0] + user_random_walk_test[1], user_random_walk_test[1] + user_random_walk_test[0]]
    # ).to(device).to(torch.int64)
    # data_test['item', 'related_to', 'item'].edge_index = torch.tensor(
    #     [item_random_walk_test[0] + item_random_walk_test[1], item_random_walk_test[1] + item_random_walk_test[0]]
    # ).to(device).to(torch.int64)
    data_test['item', 'same_store', 'item'].edge_index = torch.tensor(
        [same_store_edge_test[0] + same_store_edge_test[1], same_store_edge_test[1] + same_store_edge_test[0]]
    ).to(device).to(torch.int64)

    # print('test edge data finished')

    # Building Heterogeneous graph
    class HeteroGNN(torch.nn.Module):
        def __init__(self, num_users, hidden_channels, item_features_dim):
            super(HeteroGNN, self).__init__()
            uid_num = list(range(num_users))
            self.user_embedding = []

            for uid in uid_num:
                user_id = [k for k, v in uid_train.items() if v == uid][0]
                self.user_embedding.append(user_embeddings[user_id])

            self.conv1 = HeteroConv({
                ('user', 'rates', 'item'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('item', 'rated_by', 'user'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('user', 'buys', 'item'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('item', 'bought_by', 'user'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('user', 'review', 'item'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('item', 'reviewed_by', 'user'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('item', 'related_to', 'item'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('user', 'related_to', 'user'): SAGEConv((item_features_dim, item_features_dim), hidden_channels),
                ('item', 'same_store', 'item'): SAGEConv((item_features_dim, item_features_dim), hidden_channels)
            }, aggr='sum')
            # self.conv2 = HeteroConv({
            #     ('user', 'rates', 'item'): SAGEConv(hidden_channels, hidden_channels),
            #     ('item', 'rated_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
            #     ('user', 'buys', 'item'): SAGEConv(hidden_channels, hidden_channels),
            #     ('item', 'bought_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
            #     ('user', 'review', 'item'): SAGEConv(hidden_channels, hidden_channels),
            #     ('item', 'reviewed_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
            #     ('item', 'same_store', 'item'): SAGEConv(hidden_channels, hidden_channels)
            # }, aggr='sum')
            # self.conv3 = HeteroConv({
            #     ('user', 'rates', 'item'): SAGEConv(hidden_channels, hidden_channels),
            #     ('item', 'rated_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
            #     ('user', 'buys', 'item'): SAGEConv(hidden_channels, hidden_channels),
            #     ('item', 'bought_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
            #     ('user', 'review', 'item'): SAGEConv(hidden_channels, hidden_channels),
            #     ('item', 'reviewed_by', 'user'): SAGEConv(hidden_channels, hidden_channels),
            #     ('item', 'same_store', 'item'): SAGEConv(hidden_channels, hidden_channels)
            # }, aggr='sum')

            self.lin = Linear(hidden_channels, 1)

        def forward(self, x_dict, edge_index_dict, train_or_test):
            if train_or_test == 'train':
                # Assuming edge_index_dict is correctly formed and passed
                x_dict['user'] = torch.tensor(
                    np.stack(self.user_embedding), dtype=torch.float32, requires_grad=True, device=device
                )
                x_dict = self.conv1(x_dict, edge_index_dict)  # First layer of convolutions
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply non-linearity
            else:
                # Assuming edge_index_dict is correctly formed and passed
                x_dict['user'] = torch.tensor(np.stack(self.user_embedding), dtype=torch.float32,
                                              device=device)  # Embed user features
                x_dict = self.conv1(x_dict, edge_index_dict)  # First layer of convolutions
                x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply non-linearity
            # x_dict = self.conv2(x_dict, edge_index_dict)  # Second layer of convolutions
            # x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply non-linearity
            # x_dict = self.conv3(x_dict, edge_index_dict)
            # x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply non-linearity
            return x_dict

    # Assuming data_train and data_test are defined properly with .x, .edge_index, etc.
    num_users_train = data_train['user'].num_nodes
    num_users_test = data_test['user'].num_nodes
    item_features_dim = data_train['item'].x.size(1)

    # Instantiate the model
    model = HeteroGNN(num_users_train, hidden_channels, item_features_dim).to(device)

    # Training process
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    # Training loop
    last_epoch_loss = 0
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out_dict = model(
            {
                'user': torch.arange(num_users_train).to(device),
                'item': data_train['item'].x.to(device)
            },
            data_train.edge_index_dict,
            'train'
        )
        user_out = out_dict['user'].to(device)
        user_indices = data_train['user', 'rates', 'item'].edge_index[0]
        predicted_ratings = model.lin(user_out[user_indices]).squeeze()
        loss = criterion(predicted_ratings, data_train['user', 'rates', 'item'].edge_attr.squeeze())
        loss.backward()
        optimizer.step()

        if loss.item() < 0.05 or last_epoch_loss == loss.item():
            break

        last_epoch_loss = loss.item()
        # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        out_dict = model(
            {
                'user': torch.arange(num_users_train).to(device),
                'item': data_test['item'].x.to(device)
            },
            data_test.edge_index_dict,
            'test'
        )
        # print(data_test['item'].x)
        user_out = out_dict['user']
        # print('num_user_test', num_users_test, user_out.shape)
        user_indices = data_test['user', 'rates', 'item'].edge_index[0]
        predicted_ratings = model.lin(user_out[user_indices]).squeeze().tolist()

    # print(calculate_RMSE(predicted_ratings, testing_data['rating'].tolist()))
    # print(calculate_MAE(predicted_ratings, testing_data['rating'].tolist()))

    return predicted_ratings


def calculate_RMSE(predicted_result, true_label):
    if len(predicted_result) != len(true_label):
        return 0

    total_error = 0
    length = len(predicted_result)
    i = 0

    while i < length:
        diff = predicted_result[i] - true_label[i]
        # individual_diff.append(diff)
        total_error += (diff * diff)
        i += 1

    return np.sqrt(total_error / length)


def calculate_MAE(predicted_result, true_label):
    if len(predicted_result) != len(true_label):
        return 0

    total_error = 0
    # individual_diff = []
    length = len(predicted_result)
    i = 0

    while i < length:
        diff = predicted_result[i] - true_label[i]
        # individual_diff.append(abs(diff))
        total_error += abs(diff)
        i += 1

    return np.sqrt(total_error / length)


def get_text_embedding(item_id, item_embed_dict):
    # if item_id not in valid_item_ids:
    #     return None

    return item_embed_dict[item_id]


def get_word2vec_sentence_vector(sentence, model, vector_size):
    words = word_tokenize(sentence)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    if len(word_vectors) == 0:  # To handle cases where no words are in the model
        return np.zeros(vector_size)
    sentence_vector = np.mean(word_vectors, axis=0)
    return sentence_vector


if __name__ == '__main__':
    beauty_path = '../Datasets/beauty.pkl'
    fashion_path = '../Datasets/fashion.pkl'
    meta_beauty_path = '../Datasets/meta_beauty.pkl'
    meta_fashion_path = '../Datasets/meta_fashion.pkl'
    beauty_w2v_path = 'beauty_w2v_model.model'
    fashion_w2v_path = 'fashion_w2v_model.model'
    beauty_user_profile_path = '../Datasets/beauty_user_profile.pkl'
    fashion_user_profile_path = '../Datasets/fashion_user_profile.pkl'
    beauty_item_profile_path = '../Datasets/beauty_item_profile.pkl'
    fashion_item_profile_path = '../Datasets/fashion_item_profile.pkl'
    beauty_generated_user_embeddings = '../Datasets/beauty_generated_user_embeddings.pkl'
    fashion_generated_user_embeddings = '../Datasets/fashion_generated_user_embeddings.pkl'
    beauty_generated_item_embeddings = '../Datasets/beauty_generated_item_embeddings.pkl'
    fashion_generated_item_embeddings = '../Datasets/fashion_generated_item_embeddings.pkl'
    # sentence_model = SentenceTransformer("hkunlp/instructor-xl")
    review_instruction = "This is the review text of an Amazon User, detailed description of each key is as follows.\n\"Title\": \"The title of the product is reviewed by user\"\n\"text\": \"The review text of user\""

    num_chunks = 5
    num_folds = num_chunks
    hidden_channels = 128
    learning_rate = 0.001
    num_epochs = 1000
    # threshold_for_fashion = []
    threshold_for_fashion = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # threshold_for_fashion = [9, 10, 11, 12, 13, 14, 15, 16]
    threshold_for_beauty = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]

    device = 'cuda'
    world_size = 8
    manager = Manager()
    torch.cuda.manual_seed_all(42)  # If you're using GPU
    np.random.seed(42)

    df_path_list = [beauty_path, fashion_path]
    meta_df_path_list = [meta_beauty_path, meta_fashion_path]
    w2vec_path_list = [beauty_w2v_path, fashion_w2v_path]
    user_profile_path_list = [beauty_user_profile_path, fashion_user_profile_path]
    item_profile_path_list = [beauty_item_profile_path, fashion_item_profile_path]
    generated_user_embeddings_list = [beauty_generated_user_embeddings, fashion_generated_user_embeddings]
    generated_item_embeddings_list = [beauty_generated_item_embeddings, fashion_generated_item_embeddings]
    threshold_list = [threshold_for_beauty, threshold_for_fashion]
    count = 0

    for df_path, meta_df_path, w2vec_path, user_profile_path, item_profile_path, generated_user_embeddings, generated_item_embeddings, threshold in (
            zip(
                df_path_list,
                meta_df_path_list,
                w2vec_path_list,
                user_profile_path_list,
                item_profile_path_list,
                generated_user_embeddings_list,
                generated_item_embeddings_list,
                threshold_list
            )
    ):
        # if count == 0:
        #     count += 1
        #     continue

        RMSE_list = []
        MAE_list = []

        for local_threshold in threshold:
            df = load_dataset(df_path)
            meta_df = load_dataset(meta_df_path)
            # user_profile_dict = load_dataset(user_profile_path)
            # item_profile_dict = load_dataset(item_profile_path)
            user_embeddings = load_dataset(generated_user_embeddings)
            item_embeddings = load_dataset(generated_item_embeddings)
            valid_user_ids = list(user_embeddings.keys())
            valid_item_ids = list(item_embeddings.keys())

            # region pre-process
            df.dropna(subset=["rating"], inplace=True)

            '''
            this part is to remove empty title from interactions,
            and remove empty title from item_attributes
            '''
            item_with_empty_title = meta_df[meta_df['title'].str.strip() == '']['parent_asin'].tolist()
            meta_df = meta_df[meta_df['title'].str.strip() != '']
            df = df[~df['parent_asin'].isin(item_with_empty_title)]

            '''
            this part is to remove nan value in the store column from interactions,
            and remove nan value in the store column from item_attributes
            '''
            meta_df['store'].replace({None: np.nan})
            removed_parent_asin = meta_df.loc[meta_df['store'].isna(), 'parent_asin']
            df = df[~df['parent_asin'].isin(removed_parent_asin)]
            meta_df.dropna(subset=['store'], inplace=True)

            df = df[df['user_id'].isin(valid_user_ids) & df['parent_asin'].isin(valid_item_ids)]
            meta_df = meta_df[meta_df['parent_asin'].isin(valid_item_ids)]

            countU = df['user_id'].value_counts().to_dict()
            countP = df['parent_asin'].value_counts().to_dict()

            # Apply filtering mask based on threshold
            threshold = local_threshold
            df_mask = df['user_id'].map(countU) >= threshold
            df_mask &= df['parent_asin'].map(countP) >= threshold
            df = df[df_mask].reset_index(drop=True)

            # Create itemmap using unique parent_asins from filtered df
            valid_asins = df['parent_asin'].unique()
            itemmap = {asin: i + 1 for i, asin in enumerate(valid_asins)}  # itemid starts from 1
            itemnum = len(itemmap)

            # Filter meta_df in one line
            meta_df = meta_df[meta_df['parent_asin'].isin(itemmap)].reset_index(drop=True)

            # print('len', len(df), len(meta_df))
            # if count != 0:
            #     df = df[:100000]

            item_list = df['parent_asin'].tolist()

            # region get train, test dataset
            shuffled_data = df.sample(frac=1, random_state=42).reset_index(drop=True)
            # Split the data into 5(num_chunks) equal parts
            chunks = np.array_split(shuffled_data, num_chunks)
            # endregion

            items_dict = {}
            item_count = len(meta_df)
            counter = 0
            print_counter = 20000
            rate_count_train = len(meta_df)
            for index, row in meta_df.iterrows():
                # if counter % print_counter == 0:
                #     print(str(round(counter / rate_count_train, 2) * 100) + '%')

                items_dict[row['parent_asin']] = {
                    "average_rating": row['average_rating'],
                    "rating_number": row['rating_number'],
                    "embed": item_embeddings[row['parent_asin']],
                    "store": row['store']
                }

                counter += 1

            stores = list(set(meta_df['store'].tolist()))

            del df, meta_df, item_with_empty_title, removed_parent_asin

            mae_list = []
            rmse_list = []
            i = 0

            i = 0
            while i < num_folds:
                # Dynamically concatenate the chunks for training, excluding the one for validation
                train_chunks = []
                for j in range(num_folds - 1):  # Select (num_folds - 1) chunks for training
                    train_chunks.append(chunks[(i + j) % num_folds])

                # Concatenate all the selected chunks for training
                result = train_and_evaluate(
                    pd.concat(train_chunks),
                    chunks[(i + num_folds - 1) % num_folds],  # Validation chunk
                    items_dict,
                    user_embeddings,
                    item_embeddings
                )

                # Calculate RMSE and MAE for the validation chunk
                rmse = calculate_RMSE(result, chunks[(i + num_folds - 1) % num_folds]['rating'].tolist())
                mae = calculate_MAE(result, chunks[(i + num_folds - 1) % num_folds]['rating'].tolist())

                mae_list.append(mae)
                rmse_list.append(rmse)

                # Increment the loop counter
                i += 1

                # Clear memory
                gc.collect()
                torch.cuda.empty_cache()

            print(
                'Dataset:', df_path,
                'RMSE:', sum(rmse_list) / len(rmse_list),
                "MAE:", sum(mae_list) / len(mae_list),
                "Hidden channels:", hidden_channels,
                'threshold:', local_threshold
            )

            RMSE_list.append(round(sum(rmse_list) / len(rmse_list), 4))
            MAE_list.append(round(sum(mae_list) / len(mae_list), 4))

            print(rmse_list)
            print(mae_list)

            with open('mae.pkl', 'wb') as f:
                pickle.dump(mae_list, f)
            with open('rmse.pkl', 'wb') as f:
                pickle.dump(rmse_list, f)

            gc.collect()
            torch.cuda.empty_cache()
            # count += 1

        temp = [
            ['RMSE'] + RMSE_list,
            ['MAE'] + MAE_list
        ]

        # Create DataFrame
        df = pd.DataFrame(temp)

        if 'beauty' in df_path:
            output_path = f'../Datasets/RLMRec+HeteroGNN+InteractionFilter_hidden_channels={hidden_channels}_beauty.xlsx'
        else:
            output_path = f'../Datasets/RLMRec+HeteroGNN+InteractionFilter_hidden_channels={hidden_channels}_fashion.xlsx'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_excel(
            output_path,
            index=False, header=False
        )

'''
Dataset: ../Datasets/beauty.pkl RMSE: 1.3597100297757936 MAE: 1.0294824297664669 Hidden channels: 16 threshold: 2
[1.3577782260021969, 1.3521809886261908, 1.3811112063006101, 1.322607883111101, 1.3848718448388684]
[1.0283591923492523, 1.0271285695383585, 1.0285010638259826, 1.03068149676543, 1.0327418263533104]

Dataset: ../Datasets/beauty.pkl RMSE: 1.2276469885592012 MAE: 0.9733083854591806 Hidden channels: 16 threshold: 3
[1.2194882951438832, 1.2504539559232393, 1.2388522921889689, 1.2300180252019333, 1.1994223743379815]
[0.9715052176209569, 0.975910249361171, 0.9732889436941109, 0.9770537600266724, 0.9687837565929917]

Dataset: ../Datasets/beauty.pkl RMSE: 1.1342559565165036 MAE: 0.9408197791840797 Hidden channels: 16 threshold: 4
[1.1284374191793716, 1.121629458090992, 1.1388691346811552, 1.1508358251385147, 1.1315079454924857]
[0.9392307811606779, 0.938826712470736, 0.9403307730184773, 0.9424673040255297, 0.9432433252449771]

Dataset: ../Datasets/beauty.pkl RMSE: 1.05781842397258 MAE: 0.911905200736921 Hidden channels: 16 threshold: 5
[1.0493743977957086, 1.0205579058922802, 1.0669790583179748, 1.0858799326811974, 1.0663008251757389]
[0.9102214168857569, 0.8992968225901317, 0.9117202985815439, 0.9174983099177199, 0.9207891557094535]

Dataset: ../Datasets/beauty.pkl RMSE: 1.0110929727618847 MAE: 0.8931888319965454 Hidden channels: 16 threshold: 6
[1.0010741456108805, 1.0329205979979286, 1.0481785702865414, 1.0047120398720681, 0.9685795100420054]
[0.8881220132660826, 0.891805741793428, 0.9100252546613727, 0.8962460768358228, 0.8797450734260209]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9548410616735495 MAE: 0.8696399320957757 Hidden channels: 16 threshold: 7
[0.943098853686132, 0.9631586348888219, 0.956838278547278, 0.9608117143880509, 0.9502978268574642]
[0.8641124735748869, 0.8764889402486105, 0.8748704865613907, 0.8608874293159428, 0.8718403307780483]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9169273150121423 MAE: 0.8577679354640486 Hidden channels: 16 threshold: 8
[0.921748713360909, 0.8947267487671964, 0.9484350555026507, 0.861792820712808, 0.9579332367171479]
[0.8668839939173296, 0.8506706653173866, 0.864274970593441, 0.8377289076741202, 0.8692811398179657]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9224624010051384 MAE: 0.8577468988700305 Hidden channels: 16 threshold: 9
[0.9517485210135525, 0.983683830388724, 0.7801227183102122, 0.9140949260646652, 0.9826620092485385]
[0.8578141546879852, 0.8774245649550487, 0.8152852917855528, 0.8632161936729522, 0.8749942892486139]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9139338087993567 MAE: 0.8505744606192556 Hidden channels: 16 threshold: 10
[1.0083233482674112, 0.9635467533854316, 0.9177767063940059, 0.8396298760391023, 0.8403923599108323]
[0.8793084755520048, 0.871580189223312, 0.8448729307886969, 0.8264594145770948, 0.8306512929551696]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9207178204722769 MAE: 0.8603348914957953 Hidden channels: 16 threshold: 11
[0.9483205438765351, 0.8681659209635235, 0.8949085717745157, 0.9581773542386111, 0.9340167115081995]
[0.8680945592431449, 0.8428764568036551, 0.8502850139832039, 0.8786680560977886, 0.8617503713511842]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9261575458756438 MAE: 0.8580049555747437 Hidden channels: 16 threshold: 12
[0.8735383760421613, 0.9112637569100503, 0.9552710731387168, 0.9488285129788521, 0.9418860103084382]
[0.8547364307485192, 0.8576714010811757, 0.8541794417817775, 0.8589434284674814, 0.8644940757947648]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9069157582609474 MAE: 0.8514311127945371 Hidden channels: 16 threshold: 13
[0.929349760283481, 0.9387691782486173, 0.8080166272628896, 0.9668058186673383, 0.8916374068424111]
[0.8561707654178426, 0.8686574048252744, 0.8126987156682141, 0.867972003453147, 0.8516566746082073]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9177954202979078 MAE: 0.8568443999989462 Hidden channels: 16 threshold: 14
[0.7880449920828465, 0.8648088091806136, 0.951477925959589, 1.053834250914627, 0.9308111233518636]
[0.8088829346698674, 0.8315893598740881, 0.8809926835379908, 0.9063945379085974, 0.8563624840041877]

Dataset: ../Datasets/beauty.pkl RMSE: 0.8908387948500323 MAE: 0.8424310844838818 Hidden channels: 16 threshold: 15
[0.953599141000412, 0.9124940298094172, 0.7443105771595455, 0.9129318146384962, 0.9308584116422904]
[0.855730828799731, 0.8430531363949934, 0.8030580874084037, 0.8580856779869482, 0.8522276918293324]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9300047805677588 MAE: 0.8664324742087357 Hidden channels: 16 threshold: 16
[0.9765016457723754, 0.9238238746943293, 0.7954493906229094, 1.0116210417090112, 0.9426279500401689]
[0.8880044481034906, 0.8367181555878606, 0.8276216006760184, 0.8894881368293882, 0.8903300298469208]

Dataset: ../Datasets/beauty.pkl RMSE: 1.4089795349112384 MAE: 1.0356416912546316 Hidden channels: 32 threshold: 2
[1.4232533271484757, 1.4090354988189635, 1.4054364683045253, 1.4052504005646045, 1.4019219797196234]
[1.0399729203264307, 1.0343208473773438, 1.0341531575428236, 1.0335073424329644, 1.0362541885935956]

Dataset: ../Datasets/beauty.pkl RMSE: 1.2381670995032614 MAE: 0.9738403809108883 Hidden channels: 32 threshold: 3
[1.2549510742496897, 1.2261603696629964, 1.2428226011535273, 1.256889954404955, 1.2100114980451395]
[0.9744122023149098, 0.9735647015306319, 0.9734166547939852, 0.9794251215337137, 0.9683832243812011]

Dataset: ../Datasets/beauty.pkl RMSE: 1.154694017874451 MAE: 0.9424013546334594 Hidden channels: 32 threshold: 4
[1.1604883023336192, 1.1416911972010104, 1.1499718908995364, 1.1574898049550584, 1.1638288939830304]
[0.9417813727013232, 0.9415564525892809, 0.9407603510233364, 0.9420558226861487, 0.9458527741672075]

Dataset: ../Datasets/beauty.pkl RMSE: 1.069919009630334 MAE: 0.9123623131114773 Hidden channels: 32 threshold: 5
[1.062583665577783, 1.0343449634610393, 1.0677981706881468, 1.1087196852517391, 1.0761485631729626]
[0.9082122101914876, 0.8980235296642578, 0.9115401018819667, 0.9214022351961875, 0.9226334886234864]

Dataset: ../Datasets/beauty.pkl RMSE: 1.0241650557415387 MAE: 0.8960640983029349 Hidden channels: 32 threshold: 6
[1.0026094469007842, 1.0552420011388057, 1.0576400498235838, 1.0087687718482874, 0.9965650089962323]
[0.8883831557869926, 0.8961934297744726, 0.9131521484081011, 0.8964572967507779, 0.8861344607943302]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9648236341054922 MAE: 0.8704676121393898 Hidden channels: 32 threshold: 7
[0.958609199374878, 0.9664303384411334, 0.9757799450166871, 0.961282583945547, 0.9620161037492156]
[0.8675358989404284, 0.8764760757180271, 0.8761382463917752, 0.8574385030329005, 0.8747493366138177]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9212529593215167 MAE: 0.8563901994566304 Hidden channels: 32 threshold: 8
[0.9337343833587736, 0.8967372007453417, 0.9553711827998816, 0.8659143331560704, 0.9545076965475165]
[0.8624459969446266, 0.851199882060105, 0.8618693859672263, 0.838355027604947, 0.8680807047062473]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9254426376623422 MAE: 0.8588294099498466 Hidden channels: 32 threshold: 9
[0.960074046327266, 0.9901954005290597, 0.7622590015508974, 0.9277285171711493, 0.9869562227333387]
[0.8599857326233997, 0.8776430067005556, 0.81481224664915, 0.8669683423396716, 0.8747377214364558]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9157713877226659 MAE: 0.8504192666294685 Hidden channels: 32 threshold: 10
[1.009878913843525, 0.9661346477743338, 0.9164879379862082, 0.8431073941264167, 0.8432480448828462]
[0.8794435461604203, 0.8707523810302081, 0.8448930289016049, 0.8264696806780981, 0.8305376963770104]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9259786768869901 MAE: 0.861413294530226 Hidden channels: 32 threshold: 11
[0.9527040673077778, 0.883841370378231, 0.8946681539382286, 0.962264972326876, 0.9364148204838373]
[0.8692643417678896, 0.8459547201409581, 0.8501243446399576, 0.8793472382555869, 0.8623758278467378]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9368988325914478 MAE: 0.8600578314746302 Hidden channels: 32 threshold: 12
[0.8766136609167704, 0.9151922704400913, 0.9545818430853582, 0.9647240799464437, 0.9733823085685754]
[0.8554114615305227, 0.8580612194792978, 0.8541057697265543, 0.862969542960622, 0.8697411636761548]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9256890512215339 MAE: 0.8569557095360473 Hidden channels: 32 threshold: 13
[0.9447776906983524, 0.9796644767395621, 0.8160229526027934, 0.9733052162539568, 0.9146749198130046]
[0.8619693116795306, 0.8834613363468772, 0.8145014569147977, 0.8677387192753727, 0.8571077234636579]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9221817573424375 MAE: 0.8535198206288669 Hidden channels: 32 threshold: 14
[0.7915660089662396, 0.9031946371414068, 0.9354966759417175, 1.052136999424356, 0.928514465238467]
[0.8071524973129849, 0.8269536098171727, 0.8768011107786505, 0.9015743804857067, 0.8551175047498193]

Dataset: ../Datasets/beauty.pkl RMSE: 0.8944099819642972 MAE: 0.8420804680682945 Hidden channels: 32 threshold: 15
[0.9573040371810227, 0.917580722899029, 0.738644632018605, 0.9175096676675347, 0.9410108500552944]
[0.8561933799058732, 0.8431266958250678, 0.8020205476116801, 0.8555782862067495, 0.8534834307921025]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9495243828545459 MAE: 0.8718735700138793 Hidden channels: 32 threshold: 16
[0.9790756897521482, 0.9468822197313307, 0.8123247697939534, 1.0337308474576063, 0.9756083875376899]
[0.8887411203074662, 0.8474176272794575, 0.8238369719821856, 0.8934312246959024, 0.9059409058043852]

Dataset: ../Datasets/beauty.pkl RMSE: 1.1691649694475268 MAE: 0.9436893402863472 Hidden channels: 64 threshold: 4
[1.170845485255588, 1.163388448997131, 1.1728512750833973, 1.165552924151099, 1.1731867137504193]
[0.9429945109965162, 0.9451007323356568, 0.94226354606534, 0.941996746533824, 0.9460911655003984]

Dataset: ../Datasets/beauty.pkl RMSE: 1.081379058152761 MAE: 0.9126953495700991 Hidden channels: 64 threshold: 5
[1.0701902330557846, 1.0395102861375354, 1.084810117427487, 1.1184139083570055, 1.0939707457859915]
[0.9073824073810132, 0.898228980896058, 0.9107499829552497, 0.9228743274306993, 0.9242410491874762]

Dataset: ../Datasets/beauty.pkl RMSE: 1.0358060581979474 MAE: 0.8973968595182829 Hidden channels: 64 threshold: 6
[1.0215893807869658, 1.0589992290480486, 1.0719923697915779, 1.0234441400008019, 1.0030051713623427]
[0.8895265406315502, 0.8971473736844616, 0.9164907379274377, 0.8962841579096799, 0.8875354874382849]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9668600371846312 MAE: 0.8706515604224858 Hidden channels: 64 threshold: 7
[0.9594741337138948, 0.9709206100756457, 0.9729216703370669, 0.962587072907444, 0.9683966988891043]
[0.8677299785174116, 0.8767605747872637, 0.8758659966036503, 0.8570244941259073, 0.875876758078196]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9313168562796447 MAE: 0.8580248497772832 Hidden channels: 64 threshold: 8
[0.9474665035460254, 0.9074937642304307, 0.9651463328148939, 0.8732046160057533, 0.9632730648011198]
[0.8639740804668821, 0.8540102759722579, 0.8620419315671053, 0.8395079306756746, 0.870590030204496]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9367261032134484 MAE: 0.8603438784189285 Hidden channels: 64 threshold: 9
[0.9680739855195961, 0.9916952102608817, 0.8027936275139543, 0.9323926153584465, 0.9886750774143633]
[0.8623094509437705, 0.8777179510593227, 0.8184506274490136, 0.8683464401761112, 0.8748949224664241]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9214311222445764 MAE: 0.8507291961069459 Hidden channels: 64 threshold: 10
[1.014000056933772, 0.9745031081640104, 0.9265197302588287, 0.8485539672497538, 0.843578748616517]
[0.8796004957728413, 0.8721397316577735, 0.8444179325727239, 0.8270956214809345, 0.8303921990504567]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9400147011855979 MAE: 0.8642701790529175 Hidden channels: 64 threshold: 11
[0.9798309107148583, 0.9021744826721845, 0.9007146167045659, 0.9663237748622857, 0.9510297209740954]
[0.8761525031135713, 0.8493224875177791, 0.8507895234096464, 0.8799565174255929, 0.8651298637979974]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9541349276037797 MAE: 0.8641185739168151 Hidden channels: 64 threshold: 12
[0.9042964036510469, 0.9297121263996198, 0.9797175469596268, 0.979920608825099, 0.9770279521835061]
[0.862486029660891, 0.8605217381452605, 0.8580295924289907, 0.8692655491359, 0.870289960213033]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9418243045013762 MAE: 0.8631367647217261 Hidden channels: 64 threshold: 13
[0.9700403928474222, 1.0077866057868343, 0.8346942143063015, 0.981323945359843, 0.9152763642064803]
[0.8761594561913534, 0.8927767238709876, 0.8209694066034042, 0.8687217073765076, 0.8570565295663771]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9341275505289083 MAE: 0.8565907219112168 Hidden channels: 64 threshold: 14
[0.801374055214861, 0.9149103323298494, 0.9524840334641658, 1.0523610533636882, 0.9495082782719769]
[0.8085725553768245, 0.8305311948321107, 0.8814726587338549, 0.8947767237180928, 0.8676004768952014]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9108525217230905 MAE: 0.8450590906600504 Hidden channels: 64 threshold: 15
[0.9834010073032325, 0.9271688500542966, 0.7731663964999764, 0.9263798602282957, 0.9441464945296516]
[0.8628033421585596, 0.8447035447812035, 0.810542884185065, 0.8519773426904446, 0.8552683394849786]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9691569934493082 MAE: 0.8774542115232158 Hidden channels: 64 threshold: 16
[1.0072706527040185, 0.9696244995294103, 0.8255652504851283, 1.0375804406716531, 1.0057441238563307]
[0.898302056190779, 0.8544862844199959, 0.8262984858833501, 0.8936998030808337, 0.9144844280411202]

Dataset: ../Datasets/beauty.pkl RMSE: 1.1758223225843707 MAE: 0.9443211301463752 Hidden channels: 128 threshold: 4
[1.1743207101158308, 1.167784122923228, 1.1852439891999464, 1.1731965061945395, 1.178566284488309]
[0.9433726547554298, 0.9460836749380396, 0.9437584999647365, 0.9421198889502767, 0.946270932123393]

Dataset: ../Datasets/beauty.pkl RMSE: 1.0872271036608623 MAE: 0.9133937204013453 Hidden channels: 128 threshold: 5
[1.073771783785431, 1.0489761016327075, 1.0907516463080793, 1.1232603079002448, 1.0993756786778495]
[0.907510484011702, 0.9000381199408357, 0.9119103100187415, 0.9238760354390736, 0.9236336525963741]

Dataset: ../Datasets/beauty.pkl RMSE: 1.0472438205592076 MAE: 0.8999940778663535 Hidden channels: 128 threshold: 6
[1.0253415024884303, 1.0672408388720478, 1.0845907904148684, 1.0273290878096082, 1.0317168832110837]
[0.8896329594170224, 0.8986792841962159, 0.9183995399233303, 0.8963682577386216, 0.8968903480565772]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9750348759175301 MAE: 0.8713977701046163 Hidden channels: 128 threshold: 7
[0.9670653241933117, 0.9820793758664392, 0.9863911081645498, 0.9663945768712827, 0.9732439944920672]
[0.869581657220691, 0.8776177837016664, 0.8772610622223814, 0.8556186316597694, 0.8769097157185739]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9438991158102799 MAE: 0.860859242279526 Hidden channels: 128 threshold: 8
[0.9607910615994705, 0.9259064808359778, 0.9752325615683352, 0.8872521337519461, 0.9703133412956693]
[0.866167141941636, 0.8600017307212382, 0.8627518357076635, 0.8422738773510456, 0.8731016256760464]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9432989065741936 MAE: 0.8620690177193124 Hidden channels: 128 threshold: 9
[0.9745905813533959, 0.9998635593937457, 0.8054698318907415, 0.9433599420248018, 0.9932106182082823]
[0.8647026320813871, 0.8797591833451981, 0.8188646168647489, 0.8725004052850818, 0.8745182510201461]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9293773770462058 MAE: 0.8518418589393978 Hidden channels: 128 threshold: 10
[1.0191373262750476, 0.9829608806328604, 0.9322870182225926, 0.8587866710623528, 0.8537149890381756]
[0.879767449036551, 0.8747073453910047, 0.844057442634054, 0.8302610590225905, 0.8304159986127881]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9558033100969489 MAE: 0.868219771216799 Hidden channels: 128 threshold: 11
[1.002372559594165, 0.922864473087968, 0.9118394560243259, 0.9761295966982781, 0.9658104650800077]
[0.8828755871827056, 0.8542019273183841, 0.8538575165110619, 0.8821093800594489, 0.8680544450123949]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9683885502883882 MAE: 0.8682820097071584 Hidden channels: 128 threshold: 12
[0.9276573086463823, 0.9346552553578417, 1.0018245930958323, 0.9927242215200383, 0.9850813728218466]
[0.8691794596899829, 0.8615397608647355, 0.8647187785395801, 0.8747902583047064, 0.8711817911367867]

Dataset: ../Datasets/beauty.pkl RMSE: 0.960070516177654 MAE: 0.8691856851186415 Hidden channels: 128 threshold: 13
[0.9776623899149522, 1.0305639930400547, 0.8590456795424336, 0.986846520797393, 0.9462339975934364]
[0.8807997234056036, 0.9009545108148784, 0.82878340954768, 0.8704093660938643, 0.8649814157311814]

Dataset: ../Datasets/beauty.pkl RMSE: 0.95231078516709 MAE: 0.8624939489978363 Hidden channels: 128 threshold: 14
[0.8100520457982461, 0.9527177548822222, 0.9811421453853673, 1.0552222163274676, 0.9624197634421477]
[0.8115067047376496, 0.8439987353556948, 0.8852747070443704, 0.8949902098876923, 0.8766993879637744]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9251102496372686 MAE: 0.8507235327615426 Hidden channels: 128 threshold: 15
[1.0021480989352993, 0.9540088770764932, 0.7845193347249816, 0.9313812510534447, 0.9534936863961242]
[0.8698589341361482, 0.8538449996501983, 0.8147424463730812, 0.8535550017548721, 0.8616162818934137]

Dataset: ../Datasets/beauty.pkl RMSE: 0.9920126820341494 MAE: 0.8838041720731589 Hidden channels: 128 threshold: 16
[1.033256354364971, 0.9774560045343105, 0.8691216151511463, 1.0502649727352797, 1.0299644633850393]
[0.9078349750406799, 0.8579732330316241, 0.8407566663745616, 0.8958766636440334, 0.9165793222748948]


'''

'''
Dataset: ../Datasets/fashion.pkl RMSE: 1.2320873648199944 MAE: 0.9881336263355202 Hidden channels: 16 threshold: 4
[1.231655870890457, 1.2233717473565968, 1.2425624115097518, 1.2169634581536568, 1.2458833361895094]
[0.9884800350182498, 0.9851372345931934, 0.9906375824484519, 0.9852579854673966, 0.9911552941503092]

Dataset: ../Datasets/fashion.pkl RMSE: 1.2023360057584382 MAE: 0.9749533923715259 Hidden channels: 16 threshold: 5
[1.203458678481429, 1.2083192679952, 1.2054526849443867, 1.2167277458462338, 1.1777216515249413]
[0.9747493653113287, 0.9764093367014024, 0.9762817117619521, 0.9795859831805551, 0.967740564902391]

Dataset: ../Datasets/fashion.pkl RMSE: 1.172573544902535 MAE: 0.9633933369516093 Hidden channels: 16 threshold: 6
[1.1806782636600786, 1.1904260735601229, 1.1728485956690557, 1.1703593370523901, 1.148555454571027]
[0.9671415961576525, 0.9702200672030369, 0.9621374838313606, 0.9617154424974489, 0.9557520950685479]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1499184982258916 MAE: 0.9519793183134466 Hidden channels: 16 threshold: 7
[1.1289919389101943, 1.1617494528277539, 1.1608072575127557, 1.1484470302367153, 1.1495968116420388]
[0.9463945737635662, 0.957527201052092, 0.9555291049471727, 0.9547628726463592, 0.9456828391580424]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1198244748403074 MAE: 0.9395990845038769 Hidden channels: 16 threshold: 8
[1.1355279144304717, 1.098991840630385, 1.132298967973391, 1.1084117762709285, 1.1238918748963613]
[0.9489032752619078, 0.930177227655741, 0.9409278128214407, 0.9363062758557624, 0.9416808309245326]

Dataset: ../Datasets/fashion.pkl RMSE: 1.094853261142025 MAE: 0.9305186930533986 Hidden channels: 16 threshold: 9
[1.0770512283639628, 1.0942111627580904, 1.0992438785631609, 1.0967629346261667, 1.1069971013987445]
[0.9233235825343445, 0.9247827386435594, 0.9353982813148419, 0.9309621729248022, 0.9381266898494454]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0924861069948828 MAE: 0.9267760472254298 Hidden channels: 16 threshold: 10
[1.074634384716115, 1.069723655201478, 1.1092127085456984, 1.107965792388578, 1.1008939941225457]
[0.9214697927062802, 0.9222728089795669, 0.9365659278830213, 0.9263647079262243, 0.9272069986320566]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0743159139852863 MAE: 0.9177155041718741 Hidden channels: 16 threshold: 11
[1.1194533824660964, 1.1002395054518115, 1.0393317898524987, 1.0163181548168172, 1.0962367373392068]
[0.9270370315469995, 0.9327424623151673, 0.9012158602580768, 0.8981030976730188, 0.9294790690661086]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0603750183623002 MAE: 0.9107326528000635 Hidden channels: 16 threshold: 12
[1.098994185782935, 1.0707350770243171, 1.0725738726621867, 1.068757375350372, 0.9908145809916907]
[0.9203732056989462, 0.91121851750639, 0.9213341410731395, 0.916975645392279, 0.8837617543295634]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0903578187626306 MAE: 0.9198792956641016 Hidden channels: 16 threshold: 13
[1.0571584929572846, 1.0647644489851193, 1.1011226150850058, 1.1730376955762658, 1.055705841209477]
[0.9074437621456227, 0.9101417418239068, 0.9186034521041614, 0.9533708595971512, 0.9098366626496664]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0657678287259333 MAE: 0.9112103701073353 Hidden channels: 16 threshold: 14
[1.0739197942471015, 1.0450955454126238, 1.069317520192481, 1.0995240430352986, 1.0409822407421616]
[0.9176655560530164, 0.9104078833126336, 0.9101696602581054, 0.914205383158104, 0.9036033677548173]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0939045505870935 MAE: 0.9242384128978738 Hidden channels: 16 threshold: 15
[1.1425468062988133, 1.057731462381198, 1.0596558869899997, 1.0810613890985177, 1.1285272081669377]
[0.9369128103759241, 0.9124408513110468, 0.9138995048182404, 0.9226020182364212, 0.9353368797477368]

Dataset: ../Datasets/fashion.pkl RMSE: 1.102259417018353 MAE: 0.9255486230344309 Hidden channels: 16 threshold: 16
[1.1018308619311918, 1.0421528482131872, 1.1089507493719075, 1.1746936489641235, 1.0836689766113552]
[0.9247797115900088, 0.9167575620348233, 0.9363857594874492, 0.9393488308308178, 0.9104712512290551]

Dataset: ../Datasets/fashion.pkl RMSE: 1.2562633636995824 MAE: 0.9899768196388171 Hidden channels: 32 threshold: 4
[1.259648498714278, 1.2490235778492318, 1.2564398896599858, 1.2528962524772436, 1.2633085997971725]
[0.9895277752684045, 0.9871355514422703, 0.9911532595812195, 0.98827313778508, 0.9937943741171118]

Dataset: ../Datasets/fashion.pkl RMSE: 1.2086074238207565 MAE: 0.9752121663108954 Hidden channels: 32 threshold: 5
[1.2081252186126084, 1.2051409338866148, 1.2121745757719853, 1.228386493352648, 1.189209897479927]
[0.9746587295108603, 0.9760157121855028, 0.9766214352386325, 0.9802632559494353, 0.9685016986700455]

Dataset: ../Datasets/fashion.pkl RMSE: 1.184404123737129 MAE: 0.9641696344511047 Hidden channels: 32 threshold: 6
[1.194810562823749, 1.1980326739227296, 1.1880378699384804, 1.1845477758107534, 1.156591736189933]
[0.9685837119655164, 0.9715238455763727, 0.9625135581778607, 0.9620140202785252, 0.956213036257249]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1559529950471634 MAE: 0.9520490001403579 Hidden channels: 32 threshold: 7
[1.1422190157641634, 1.1681941972755059, 1.161366026126247, 1.1570853190828567, 1.150900416987044]
[0.945810357846262, 0.9580216682624149, 0.9544931962072106, 0.9557282360640279, 0.9461915423218734]

Dataset: ../Datasets/fashion.pkl RMSE: 1.125164964851614 MAE: 0.9401082261212178 Hidden channels: 32 threshold: 8
[1.140137133612572, 1.1071814378752944, 1.135985414744397, 1.1090626378659854, 1.1334582001598206]
[0.9507701664464543, 0.9314282306256703, 0.9402052694666307, 0.9364126499192758, 0.9417248141480585]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0960250030865637 MAE: 0.9293930300942789 Hidden channels: 32 threshold: 9
[1.0858889824313243, 1.095185835016731, 1.099197602115175, 1.0977743484677023, 1.1020782474018864]
[0.9248558825682314, 0.9247363074281353, 0.9323280282323158, 0.9299182985707345, 0.9351266336719775]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0941582454425813 MAE: 0.9269548713849642 Hidden channels: 32 threshold: 10
[1.076707199383608, 1.0749591402857779, 1.1089876146031001, 1.110474847049626, 1.0996624258907945]
[0.9202726312065864, 0.9262506234878901, 0.9369171070615738, 0.9266742915414959, 0.9246597036272748]

Dataset: ../Datasets/fashion.pkl RMSE: 1.078019763923379 MAE: 0.9177471457698477 Hidden channels: 32 threshold: 11
[1.1193390236801182, 1.102558128570122, 1.042038096481295, 1.0254195430265673, 1.1007440278587932]
[0.9268137747898764, 0.930556744682286, 0.9011691960136223, 0.9004074196863138, 0.9297885936771405]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0649859628606309 MAE: 0.9098133470435441 Hidden channels: 32 threshold: 12
[1.1014717418920394, 1.077341583614636, 1.0741131000167905, 1.0746728642253582, 0.9973305245543298]
[0.9188159991582335, 0.9100930414677714, 0.9207960762301254, 0.9172980913510895, 0.8820635270105004]

Dataset: ../Datasets/fashion.pkl RMSE: 1.091720716141564 MAE: 0.9187744595334717 Hidden channels: 32 threshold: 13
[1.0593936852971497, 1.0667778511933983, 1.1006965296557905, 1.178929895297992, 1.0528056192634894]
[0.9065836148722412, 0.9099393829769945, 0.9183229665950317, 0.954380714795591, 0.9046456184275]

Dataset: ../Datasets/fashion.pkl RMSE: 1.07174963172575 MAE: 0.9101368241516766 Hidden channels: 32 threshold: 14
[1.073990848861444, 1.044824064743327, 1.0614306502921784, 1.1099568934413595, 1.0685457012904411]
[0.9160292200482505, 0.9060506828448486, 0.9070747193012397, 0.9156943422552224, 0.9058351563088229]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0988346690299013 MAE: 0.9219698295521861 Hidden channels: 32 threshold: 15
[1.1461033848955158, 1.0648721164067696, 1.087180375309311, 1.0565198625991952, 1.139497605938716]
[0.9378256340220237, 0.8963677103323353, 0.9251139078759857, 0.9118143205283655, 0.9387275750022199]


Dataset: ../Datasets/fashion.pkl RMSE: 1.1232801240541985 MAE: 0.9292682211926555 Hidden channels: 32 threshold: 16
[1.148691855779066, 1.0572888331505184, 1.1276681624446323, 1.1974082611629415, 1.0853435077338345]
[0.9353632985504207, 0.919600133657425, 0.9409964829495807, 0.9394114995620723, 0.9109696912437791]

Dataset: ../Datasets/fashion.pkl RMSE: 1.2632421325688588 MAE: 0.9909145676859138 Hidden channels: 64 threshold: 4
[1.2615235426127498, 1.2568855764384799, 1.2679478033285319, 1.258378209365515, 1.2714755310990178]
[0.9897710793497208, 0.9885592900507686, 0.9920697151990954, 0.9890890645147458, 0.9950836893152385]

Dataset: ../Datasets/fashion.pkl RMSE: 1.2219469454906182 MAE: 0.9760955351965499 Hidden channels: 64 threshold: 5
[1.219652159267739, 1.2165668541559018, 1.2228129180363452, 1.241938428540574, 1.2087643674525306]
[0.9750603601953205, 0.9767555013418786, 0.9775594324599717, 0.9813901191963688, 0.9697122627892107]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1961476912421598 MAE: 0.9653292096782773 Hidden channels: 64 threshold: 6
[1.2151268129447463, 1.2142995837681154, 1.1915227701392423, 1.1901555389061345, 1.1696337504525611]
[0.9706690762645095, 0.9734973416511234, 0.962989584128781, 0.9623449078919469, 0.9571451384550255]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1620805467746476 MAE: 0.9527059059645939 Hidden channels: 64 threshold: 7
[1.1448597876849183, 1.17370148172936, 1.170122363674188, 1.159659064850724, 1.1620600359340474]
[0.9460505075796328, 0.9583718533369883, 0.9535234155195728, 0.9561450949210237, 0.9494386584657524]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1331100532747238 MAE: 0.9417547560964994 Hidden channels: 64 threshold: 8
[1.1496997133400833, 1.1183432469653287, 1.141923705319764, 1.115921642907805, 1.139661957840638]
[0.9541087650496308, 0.933462853920268, 0.9408144905976669, 0.9380257965071505, 0.9423618744077812]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0999108156814998 MAE: 0.9294232878504477 Hidden channels: 64 threshold: 9
[1.0897050466322407, 1.1047728645623371, 1.1027272667208945, 1.097557513763697, 1.1047913867283303]
[0.9254087113450299, 0.9258307356522615, 0.931191293131262, 0.9299869190961975, 0.9346987800274875]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1024618526811472 MAE: 0.9284072083272046 Hidden channels: 64 threshold: 10
[1.090476503643398, 1.0865382142171522, 1.1138839102497142, 1.118892556808743, 1.1025180784867281]
[0.92197479718908, 0.9305267423252332, 0.9391453328992024, 0.9274824251920363, 0.9229067440304711]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0852800711356445 MAE: 0.9183978561602688 Hidden channels: 64 threshold: 11
[1.1250707362013193, 1.1076403522915397, 1.0490181503980398, 1.0438015940479495, 1.1008695227393734]
[0.9248906441154335, 0.9297546165009685, 0.9013807924008562, 0.9063372373105447, 0.9296259904735409]

Dataset: ../Datasets/fashion.pkl RMSE: 1.070711372920702 MAE: 0.9100991584940624 Hidden channels: 64 threshold: 12
[1.1069113605694996, 1.0807767591418127, 1.0793368143136057, 1.082567193247272, 1.0039647373313199]
[0.9194110114372777, 0.9109568407276183, 0.9209784817586096, 0.9178183133722311, 0.8813311451745758]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1040789923861871 MAE: 0.9203368093834333 Hidden channels: 64 threshold: 13
[1.0760515204245527, 1.092230627119596, 1.1099383885373697, 1.19075907129513, 1.051415354554287]
[0.9072143151427823, 0.9132136304920467, 0.9188687153755559, 0.9568829966209628, 0.9055043892858189]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0856851308234148 MAE: 0.913047143867287 Hidden channels: 64 threshold: 14
[1.0848676611353454, 1.0514659961431518, 1.0606452601650203, 1.1492258360979906, 1.0822209005755663]
[0.917619261905583, 0.9051993649963703, 0.906778107126067, 0.9252774327691429, 0.9103615525392724]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1162688081598957 MAE: 0.926435339801235 Hidden channels: 64 threshold: 15
[1.1704267630713145, 1.095458936759239, 1.1167715835555636, 1.0526030539940145, 1.1460837034193456]
[0.9467125516280634, 0.9000083519991332, 0.9343895348777074, 0.9097021281143519, 0.9413641323869189]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1516811954887392 MAE: 0.9398974521194352 Hidden channels: 64 threshold: 16
[1.1618010192742563, 1.1063382596882385, 1.1479223658202893, 1.214201534068089, 1.128142798592823]
[0.9399106302621159, 0.9363556507170735, 0.9484865879224088, 0.9463876616560644, 0.9283467300395133]

Dataset: ../Datasets/fashion.pkl RMSE: 1.266426868828788 MAE: 0.991333892334197 Hidden channels: 128 threshold: 4
[1.2660477249210758, 1.2608358643611373, 1.2689203669900821, 1.2612847745158964, 1.2750456133557484]
[0.9899727289362297, 0.9893758414655611, 0.9921424986225821, 0.9895237156934199, 0.9956546769531925]

Dataset: ../Datasets/fashion.pkl RMSE: 1.225450320481835 MAE: 0.9762868806447751 Hidden channels: 128 threshold: 5
[1.2223278981790235, 1.2245711706659554, 1.2262717622904131, 1.2441316972164334, 1.2099490740573495]
[0.9750822789240275, 0.9772241156098995, 0.9779146544504251, 0.9814839170003099, 0.9697294372392135]

Dataset: ../Datasets/fashion.pkl RMSE: 1.2012960009027416 MAE: 0.9658551626927958 Hidden channels: 128 threshold: 6
[1.2172208588791391, 1.2195613962201983, 1.1984453901355359, 1.194978946971232, 1.1762734123076033]
[0.9706028409062537, 0.9739248764425431, 0.963907770651611, 0.9630926496556784, 0.9577476758078924]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1678382510274001 MAE: 0.9536747655547009 Hidden channels: 128 threshold: 7
[1.1509021800006125, 1.1829128897132988, 1.1723416236763724, 1.1635163281120018, 1.1695182336347147]
[0.9471543221491188, 0.9596976116319791, 0.9534863441183882, 0.9566297718251672, 0.9514057780488506]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1385803362858764 MAE: 0.9430573228453308 Hidden channels: 128 threshold: 8
[1.1604011793851072, 1.1182844348589118, 1.1446590918130175, 1.1239990805960274, 1.145557894776318]
[0.9569753165994778, 0.9334495695370926, 0.9412872491518729, 0.9405848172034514, 0.9429896617347602]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1061579223388303 MAE: 0.9296775292239907 Hidden channels: 128 threshold: 9
[1.0975502042315135, 1.1116730932577552, 1.1088706438075613, 1.103449981123392, 1.1092456892739289]
[0.9264855941694959, 0.9271272545551665, 0.931105730545517, 0.9284725272035182, 0.9351965396462556]

Dataset: ../Datasets/fashion.pkl RMSE: 1.108183595640163 MAE: 0.9294640239334129 Hidden channels: 128 threshold: 10
[1.0943830778957135, 1.0899459568105339, 1.1210123048321625, 1.130086869779298, 1.105489768883108]
[0.9228732668557093, 0.931627755086497, 0.9409756468125738, 0.9292248342779517, 0.9226186166343326]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0962418058843224 MAE: 0.9200474017024248 Hidden channels: 128 threshold: 11
[1.1342322690879116, 1.1155217303683465, 1.0601705984447405, 1.0613513559897545, 1.109933075530859]
[0.9249005797419132, 0.9299371902301562, 0.9023910963864321, 0.9118408960458514, 0.9311672461077706]

Dataset: ../Datasets/fashion.pkl RMSE: 1.0877244531181198 MAE: 0.9141981854483643 Hidden channels: 128 threshold: 12
[1.119228278615498, 1.0965015858715916, 1.0844735596978519, 1.1138869827940308, 1.0245318586116263]
[0.923205905021413, 0.9172135295567072, 0.9212249869285876, 0.9241840751343612, 0.8851624306007525]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1268704245158232 MAE: 0.9265701697397448 Hidden channels: 128 threshold: 13
[1.102333132159731, 1.1211079321634994, 1.122712831526859, 1.2194630363421517, 1.0687351903868747]
[0.9190525202783986, 0.9228795049026426, 0.9211761066782909, 0.9663474238387367, 0.9033952930006547]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1017810835626052 MAE: 0.9175343377742055 Hidden channels: 128 threshold: 14
[1.1104123027609112, 1.064227355066661, 1.069849964748499, 1.1557980164540314, 1.1086177787829232]
[0.9249885970596732, 0.9079259606364526, 0.9093551121396217, 0.927808746682382, 0.9175932723528979]

Dataset: ../Datasets/fashion.pkl RMSE: 1.141578157086613 MAE: 0.9362996909039767 Hidden channels: 128 threshold: 15
[1.1842250382261499, 1.1116596065604578, 1.1511412818026017, 1.0683016613355565, 1.192563197508299]
[0.9526758560471806, 0.9067526484491177, 0.94387682308433, 0.9144588903451821, 0.963734236594073]

Dataset: ../Datasets/fashion.pkl RMSE: 1.1833366380689394 MAE: 0.9543127382435175 Hidden channels: 128 threshold: 16
[1.204000651798603, 1.1235652400417164, 1.1728570036412505, 1.2660480954127673, 1.1502121994503598]
[0.9625600308537714, 0.9432400335539926, 0.9598991401222002, 0.9678282746019383, 0.9380362120856849]


'''