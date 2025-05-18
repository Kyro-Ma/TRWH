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
    data_train['item', 'bought_by', 'user'].edge_idex = torch.tensor(
        [verify_buy_to_train, verify_buy_from_train]
    ).to(device)
    item_random_walk_train = random_walk(data_train['item', 'rated_by', 'user']['edge_index'])
    user_random_walk_train = random_walk(data_train['user', 'rates', 'item']['edge_index'])
    data_train['user', 'related_to', 'user'].edge_index = torch.tensor(
        [user_random_walk_train[0] + user_random_walk_train[1],
         user_random_walk_train[1] + user_random_walk_train[0]]).to(device)
    data_train['item', 'related_to', 'item'].edge_index = torch.tensor(
        [item_random_walk_train[0] + item_random_walk_train[1],
         item_random_walk_train[1] + item_random_walk_train[0]]).to(device)
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
    data_test['item', 'bought_by', 'user'].edge_idex = torch.tensor(
        [verify_buy_to_test, verify_buy_from_test]
    ).to(device).to(torch.int64)
    item_random_walk_test = random_walk(data_test['item', 'rated_by', 'user']['edge_index'])
    user_random_walk_test = random_walk(data_test['user', 'rates', 'item']['edge_index'])
    data_test['user', 'related_to', 'user'].edge_index = torch.tensor(
        [user_random_walk_test[0] + user_random_walk_test[1], user_random_walk_test[1] + user_random_walk_test[0]]
    ).to(device).to(torch.int64)
    data_test['item', 'related_to', 'item'].edge_index = torch.tensor(
        [item_random_walk_test[0] + item_random_walk_test[1], item_random_walk_test[1] + item_random_walk_test[0]]
    ).to(device).to(torch.int64)
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


def random_walk(item_user_edge):
    new_edge = [[], []]
    item = item_user_edge[0]
    user = item_user_edge[1]
    for i in range(len(item_user_edge[0])):
        # if i % 10000 == 0:
        #     print(i / len(item))
        start = item[i]
        neighbours = user[item == start]
        random_neighbour = random.choice(neighbours)
        final_items = item[user == random_neighbour]
        final_items = final_items[final_items != start]
        if (len(final_items) > 0):
            new_edge[0].append(start.tolist())
            new_edge[1].append(random.choice(final_items).tolist())

    return new_edge


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
    hidden_channels = 16
    learning_rate = 0.001
    num_epochs = 1000
    threshold_for_fashion = []
    # threshold_for_fashion = [11, 12, 13, 14, 15, 16]
    threshold_for_beauty = [7, 8, 9]
    # threshold_for_beauty = [11, 12, 13, 14, 15, 16]

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
            output_path = f'../Datasets/RLMRec+RandomWalk+HeteroGNN+InteractionFilter_hidden_channels={hidden_channels}_beauty.xlsx'
        else:
            output_path = f'../Datasets/RLMRec+RandomWalk+HeteroGNN+InteractionFilter_hidden_channels={hidden_channels}_fashion.xlsx'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_excel(
            output_path,
            index=False, header=False
        )
    '''
    RLMRec with random walk on beauty
    Dataset: ../Datasets/beauty.pkl RMSE: 1.4005840435188603 MAE: 1.040167104815675 Hidden channels: 16 threshold: 2
    [1.4293215321483925, 1.3927721883027824, 1.354725086009, 1.4111098008092768, 1.4149916103248488]
    [1.0443747802056091, 1.0359187823048859, 1.03847059288546, 1.0389912216135315, 1.0430801470688895]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.2491816348373228 MAE: 0.9838720913910933 Hidden channels: 16 threshold: 3
    [1.2165338597841577, 1.2623856898965176, 1.2627897936784247, 1.2686004693040036, 1.2355983615235093]
    [0.9829458276783055, 0.986837958915249, 0.9823154929844224, 0.9879358279660596, 0.9793253494114307]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.1541874892144093 MAE: 0.9505503954769067 Hidden channels: 16 threshold: 4
    [1.1455450466638049, 1.1624009917587141, 1.1572093416420488, 1.1473365257613808, 1.1584455402460987]
    [0.946413548518489, 0.9507313638180087, 0.9472974230210073, 0.9566106437036059, 0.9516989983234218]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.0794467753072934 MAE: 0.9177935842005134 Hidden channels: 16 threshold: 5
    [1.066452137828333, 1.0343617151227606, 1.087578449460259, 1.1140910723968416, 1.0947505017282724]
    [0.9146595862900595, 0.9034289914027879, 0.9173406698842799, 0.9258041713208454, 0.9277345021045948]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.0267202535691564 MAE: 0.898490570210962 Hidden channels: 16 threshold: 6
    [1.0086167763394154, 1.0573803660741843, 1.0732910467176466, 1.0099470373017816, 0.9843660414127541]
    [0.8924279496545098, 0.8989275654761736, 0.9160064881906191, 0.8991106475886348, 0.885980200144873]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9627251544317688 MAE: 0.8752341856075956 Hidden channels: 16 threshold: 7
    [0.9450379999405951, 0.9690023295954314, 0.9671411299902404, 0.964868840940777, 0.9675754716918004]
    [0.8684601055800916, 0.8832686650055325, 0.8821232018744218, 0.8652191594326978, 0.8770997961452343]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.928099382527184 MAE: 0.8605099108163221 Hidden channels: 16 threshold: 8
    [0.9384130917206259, 0.9107077912414819, 0.9550663098467165, 0.874811651324796, 0.9614980685022995]
    [0.8650124336524283, 0.8519512702970081, 0.8670745927378661, 0.8456467288371289, 0.8728645285571796]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9347581636816453 MAE: 0.8653719276930587 Hidden channels: 16 threshold: 9
    [0.9490371045787068, 0.9927910013264418, 0.8123620999034112, 0.9307172832239745, 0.9888833293756925]
    [0.8652973299143226, 0.8795995891938673, 0.8341333755883321, 0.8706603860665972, 0.8771689577021751]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.920958659156528 MAE: 0.8562416265519041 Hidden channels: 16 threshold: 10
    [1.0062911752618429, 0.9735399937120705, 0.9166502304671766, 0.8447200352887554, 0.8635918610527948]
    [0.8821088840713154, 0.8753478335605719, 0.8492929351201254, 0.8311594424828044, 0.8432990375247031]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9554954775865167 MAE: 0.8812966577088541 Hidden channels: 16 threshold: 11
    [1.010411627819475, 0.9135770571909763, 0.917864519375881, 0.9717416859569596, 0.9638824975892912]
    [0.9069920498011967, 0.8697081834317174, 0.8692323326642051, 0.884721519026042, 0.8758292036211096]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9523361254598889 MAE: 0.8735950401752429 Hidden channels: 16 threshold: 12
    [0.931957809404788, 0.9146077869162045, 0.9781402833026468, 0.9736014029782156, 0.963373344697589]
    [0.8881231897867354, 0.8654099215750953, 0.8707029037664893, 0.8697121029290362, 0.8740270828188584]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9534886256159971 MAE: 0.8795656125601339 Hidden channels: 16 threshold: 13
    [0.9785422560732897, 0.9982607366028758, 0.8482417989058664, 1.010847768364543, 0.9315505681334112]
    [0.8884315889084072, 0.8998930573073037, 0.837026857351319, 0.8984432308629619, 0.8740333283706768]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9297671788918983 MAE: 0.8631874379320703 Hidden channels: 16 threshold: 14
    [0.8025579007123737, 0.9041069983228402, 0.9511255731049982, 1.0418428511670537, 0.9492025711522255]
    [0.8127969763456705, 0.8297352651670155, 0.8881560747577478, 0.9082277556250099, 0.8770211177649079]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9178972557735297 MAE: 0.8570658972617323 Hidden channels: 16 threshold: 15
    [0.9556429028830818, 0.9235420764965857, 0.774108272860481, 0.9756022807500744, 0.9605907458774251]
    [0.8565364356339538, 0.8410355165046273, 0.8135084849851547, 0.8967494061653106, 0.8774996430196154]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9548776746863366 MAE: 0.876981588892388 Hidden channels: 16 threshold: 16
    [1.0045355828966462, 0.9506368357024725, 0.826719641985378, 1.0176385973600615, 0.9748577154871247]
    [0.8980198333599981, 0.8575923424847033, 0.8295454461000198, 0.890335373918045, 0.9094149485991738]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.4297002426644478 MAE: 1.043344328303792 Hidden channels: 32 threshold: 2
    [1.4393729650819742, 1.421531231223448, 1.410711099024376, 1.4323071266548746, 1.4445787913375658]
    [1.0468024136629845, 1.0396394383170116, 1.0382261461302802, 1.0424637809667356, 1.0495898624419484]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.2601065410148453 MAE: 0.9831875416582634 Hidden channels: 32 threshold: 3
    [1.2519968741127647, 1.276378694191085, 1.2642642737138863, 1.2669494680190605, 1.240943395037431]
    [0.9803099565947323, 0.9863570585839053, 0.9824501444720956, 0.9880637238188136, 0.97875682482177]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.1701656410793013 MAE: 0.9498350636141826 Hidden channels: 32 threshold: 4
    [1.167923347107253, 1.1654403864810505, 1.1705787403010754, 1.1702533753720663, 1.176632356135062]
    [0.9472869205618084, 0.9517323972305427, 0.9478225996578725, 0.9507955923484691, 0.9515378082722201]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.086284654894474 MAE: 0.9173052199586339 Hidden channels: 32 threshold: 5
    [1.0685860351721452, 1.0486280587035903, 1.084687952465688, 1.1271041122654952, 1.1024171158654503]
    [0.91282278879105, 0.901378762037667, 0.9165828635365044, 0.9266288948044381, 0.92911279062351]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.0382778806288406 MAE: 0.8979599324483907 Hidden channels: 32 threshold: 6
    [1.0235822092596716, 1.065235755597566, 1.080414656562486, 1.0257130579979592, 0.9964437237265202]
    [0.8891795101045196, 0.8991857393864432, 0.9180306310125773, 0.8969585970830872, 0.8864451846553262]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9683122376490774 MAE: 0.8730529063769076 Hidden channels: 32 threshold: 7
    [0.9508089974445572, 0.9766989790050413, 0.9800513320720838, 0.966570049542565, 0.9674318301811397]
    [0.8669547386364976, 0.8827946677990526, 0.8767920384092417, 0.861594069620994, 0.8771290174187526]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9334735120927325 MAE: 0.8575400790825896 Hidden channels: 32 threshold: 8
    [0.9569060688219232, 0.9108693938446645, 0.9644005271233513, 0.8716375045861011, 0.9635540660876224]
    [0.862490625350491, 0.8513410319428517, 0.8618394325654475, 0.842362592857486, 0.8696667126966715]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9363391609020905 MAE: 0.8610781014410478 Hidden channels: 32 threshold: 9
    [0.9615894419674041, 0.9930662910562976, 0.7969130300708158, 0.9404038778614185, 0.9897231635545164]
    [0.8615221737003225, 0.8776482262587232, 0.819355249549845, 0.8683771207086897, 0.878487736987658]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9212851643535368 MAE: 0.8514912109495578 Hidden channels: 32 threshold: 10
    [1.0092255468545186, 0.9869784918907488, 0.9199741792961482, 0.8452608911785505, 0.8449867125477177]
    [0.8812903860468335, 0.8725007588123628, 0.8461200307690343, 0.8250346326925175, 0.8325102464270404]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9558325804383248 MAE: 0.8753170318404772 Hidden channels: 32 threshold: 11
    [1.0063879151669048, 0.913256737708217, 0.9077268384508322, 0.9771834062776864, 0.9746080045879835]
    [0.8967791681101678, 0.8588039792553878, 0.8522825593265484, 0.8885255801646469, 0.880193872345635]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9617008747706611 MAE: 0.8705320694209562 Hidden channels: 32 threshold: 12
    [0.9404978816558205, 0.921272279135945, 0.9824237245917974, 0.9730361304252815, 0.9912743580444614]
    [0.8836658959209086, 0.8642755982799409, 0.8653686491557117, 0.8670345393537727, 0.8723156643944471]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.959257818600153 MAE: 0.8786057811366458 Hidden channels: 32 threshold: 13
    [1.0100758926705704, 1.01879086325263, 0.8493804170754637, 0.982350582620298, 0.9356913373818027]
    [0.9032852231557809, 0.9055041452164309, 0.8370384585039522, 0.8717664354133291, 0.875434643393736]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9410789587070838 MAE: 0.8616156476710595 Hidden channels: 32 threshold: 14
    [0.8141446460529158, 0.9399535305307203, 0.9556121326715777, 1.0485366078670266, 0.9471478764131783]
    [0.8196649802463449, 0.8392251448197938, 0.8651330820640865, 0.9092986599746214, 0.8747563712504511]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9133630719684511 MAE: 0.8532892578280652 Hidden channels: 32 threshold: 15
    [0.9763676802099209, 0.9181299444407818, 0.7717225510416043, 0.9350434909644675, 0.9655516931854811]
    [0.8717192009793532, 0.8410934382433305, 0.8067831227826171, 0.8666805092872861, 0.8801700178477394]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9573392576060258 MAE: 0.873744076441113 Hidden channels: 32 threshold: 16
    [1.0108115194220506, 0.9581464063281173, 0.8261573662977802, 1.020910604053747, 0.9706703919284343]
    [0.9021391519945696, 0.8553779792829429, 0.8259534549139783, 0.8862060659006007, 0.8990437301134736]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.4406236363593774 MAE: 1.0455162697763394 Hidden channels: 64 threshold: 2
    [1.4515101719154304, 1.4294098670846622, 1.436880847240802, 1.4365041210570575, 1.4488131744989354]
    [1.0492697059868155, 1.0408887635828243, 1.0433331407021917, 1.0434051694116058, 1.0506845691982594]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.2758789649555118 MAE: 0.9840706140570321 Hidden channels: 64 threshold: 3
    [1.2755557555343653, 1.282253256815088, 1.2744685896508205, 1.284749574935022, 1.2623676478422625]
    [0.9827254561085286, 0.9868403646072925, 0.9836211470197413, 0.9880725936710207, 0.9790935088785772]

    Dataset: ../Datasets/beauty.pkl RMSE: 1.1820765706675402 MAE: 0.9503192460010063 Hidden channels: 64 threshold: 4
    [1.1781692993054984, 1.1739798580266052, 1.188604675961352, 1.1837889425445811, 1.1858400774996638]
    [0.9490434761215596, 0.9525693009871518, 0.9482087257174803, 0.950996052824744, 0.9507786743540951]

    Dataset: ../Datasets/beauty.pkl RMSE: 1.0938476464517732 MAE: 0.9170183369907889 Hidden channels: 64 threshold: 5
    [1.0820691367940554, 1.0542750445752653, 1.0964315319306892, 1.1290758473102083, 1.1073866716486478]
    [0.9126760917916293, 0.9015195351545394, 0.9178720519174789, 0.9269352705505172, 0.9260887355397789]

    Dataset: ../Datasets/beauty.pkl RMSE: 1.0488611680333213 MAE: 0.8996177030227355 Hidden channels: 64 threshold: 6
    [1.0271037798772522, 1.0688987105072234, 1.0897477336021828, 1.0403438087291417, 1.0182118074508069]
    [0.8896414957350225, 0.8998339632447458, 0.9191041551598117, 0.8979418654482844, 0.8915670355258132]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9801238002479076 MAE: 0.8727011029801517 Hidden channels: 64 threshold: 7
    [0.9746796484147194, 0.9816649614027426, 0.9874834127411962, 0.9750175037953414, 0.9817734748855386]
    [0.8700655662695643, 0.8819917226082039, 0.8768427265928562, 0.8574652694787012, 0.8771402299514325]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9415708852898298 MAE: 0.8577982840081442 Hidden channels: 64 threshold: 8
    [0.9555599685229326, 0.9258171832159038, 0.9725932307810035, 0.8830518823242293, 0.9708321616050802]
    [0.8598378369661829, 0.8541628046065032, 0.8610005854726192, 0.8411314275845652, 0.87285876541085]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9428281210180927 MAE: 0.8609377119274193 Hidden channels: 64 threshold: 9
    [0.9826161900716256, 1.0002496048444032, 0.80674194302556, 0.9323916298371108, 0.992141237311764]
    [0.861225682532202, 0.8773111105670366, 0.8218703490893192, 0.8683865972218182, 0.8758948202267199]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9329682463084407 MAE: 0.8519125044829714 Hidden channels: 64 threshold: 10
    [1.0174032098307706, 1.004499753246715, 0.9352954021617479, 0.8566301492359152, 0.8510127170670544]
    [0.8798988503595658, 0.8795895690592076, 0.8446163423666266, 0.8261157615120859, 0.8293419991173706]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9616554066662288 MAE: 0.8731980670911581 Hidden channels: 64 threshold: 11
    [1.021583318183492, 0.9148641741240126, 0.9178644099654375, 0.9810312789843976, 0.9729338520738054]
    [0.8950052716044942, 0.8520183774814378, 0.8527098053827507, 0.8894852330336104, 0.876771647953497]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9644406164272151 MAE: 0.868779071275935 Hidden channels: 64 threshold: 12
    [0.9442070590086713, 0.9243894647877139, 0.9918631520901633, 0.9739458432252679, 0.98779756302426]
    [0.8850996069643404, 0.861008482878978, 0.8642388915656403, 0.8646520859300342, 0.8688962890406823]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9679533329322675 MAE: 0.8767842337337246 Hidden channels: 64 threshold: 13
    [1.006298091127052, 1.0234151371848077, 0.862935990623126, 0.9924233578875762, 0.9546940878387757]
    [0.9013899552706073, 0.8983909775402056, 0.8335519157121694, 0.8788679592708438, 0.8717203608747978]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9388456541488907 MAE: 0.8582294497090709 Hidden channels: 64 threshold: 14
    [0.8057335219221394, 0.9224371992225078, 0.9590580387675619, 1.0508087980552703, 0.956190712776974]
    [0.8106404547322377, 0.8361030781577287, 0.8657838756393464, 0.9017512152544189, 0.8768686247616226]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9217391860448837 MAE: 0.850784212330443 Hidden channels: 64 threshold: 15
    [0.9855829970713568, 0.9409125423002506, 0.8033733599654497, 0.9272521518044322, 0.9515748790829291]
    [0.8703047536439088, 0.8474999220092012, 0.8181554909782014, 0.8544122318566013, 0.8635486631643016]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9770145707800557 MAE: 0.8744524659163782 Hidden channels: 64 threshold: 16
    [1.0234717843201044, 0.9676414746254098, 0.8429680358064531, 1.0360380449353963, 1.0149535142129154]
    [0.9053507107323199, 0.8320290154856596, 0.831530767204309, 0.8970097673059048, 0.9063420688536975]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.445962834485518 MAE: 1.046842151162113 Hidden channels: 128 threshold: 2
    [1.455089504379854, 1.4363129747638117, 1.4431073034741546, 1.4430857143775684, 1.4522186754322006]
    [1.0505039395268776, 1.0425403521528378, 1.0451154241349938, 1.0448106772713754, 1.0512403627244804]

    Dataset: ../Datasets/beauty.pkl RMSE: 1.280525265180756 MAE: 0.9844396710246681 Hidden channels: 128 threshold: 3
    [1.2827921676733633, 1.2858622441368948, 1.2782827566659314, 1.2910284668006133, 1.2646606906269766]
    [0.9847874338383115, 0.9862108780020234, 0.9844687790213882, 0.9878672704060399, 0.9788639938555771]

    Dataset: ../Datasets/beauty.pkl RMSE: 1.1866143677459502 MAE: 0.9492886556107617 Hidden channels: 128 threshold: 4
    [1.1836567413447279, 1.1759943647190267, 1.1963156334208722, 1.1873455329771705, 1.1897595662679534]
    [0.9471849888136178, 0.9512765569969241, 0.9477808547189908, 0.950323373192831, 0.9498775043314454]

    Dataset: ../Datasets/beauty.pkl RMSE: 1.0977516752977778 MAE: 0.9175186170262016 Hidden channels: 128 threshold: 5
    [1.088012868292762, 1.0618421364382968, 1.09997560242111, 1.1291193932137107, 1.1098083761230089]
    [0.9128142125991042, 0.9034677310079249, 0.9191644068663931, 0.9258532668210369, 0.9262934678365491]

    Dataset: ../Datasets/beauty.pkl RMSE: 1.060957860764282 MAE: 0.9019511264746587 Hidden channels: 128 threshold: 6
    [1.0384588450482835,` 1.0784588701750495, 1.098917989646145, 1.0476010846062216, 1.0413525143457112]
    [0.8899819586423406, 0`.9012033527480708, 0.921298964240629, 0.9000026625361596, 0.8972686942060936]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9862004637261437 MAE: 0.8735002627085775 Hidden channels: 128 threshold: 7
    [0.9865511929743465, 0.9905642696232096, 0.9903591136053986, 0.9795819916971746, 0.9839457507305893]
    [0.8741329659954103, 0.8810637869881811, 0.8754155742786655, 0.8588829356767915, 0.8780060506038393]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9548404703626329 MAE: 0.8606398598727566 Hidden channels: 128 threshold: 8
    [0.9731147132128367, 0.9392603636460848, 0.985584484650191, 0.8944409113833933, 0.9818018789206587]
    [0.8648866296598967, 0.8568610449342439, 0.86286892037233, 0.8426478228065853, 0.8759348815907271]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9562942206403994 MAE: 0.861700463016895 Hidden channels: 128 threshold: 9
    [0.9935788193857396, 1.0237493780214237, 0.8138437678964988, 0.9551683175128601, 0.995130820385475]
    [0.8615706708426965, 0.8843545216369213, 0.8177429225616142, 0.8732584809354156, 0.8715757191078277]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9426176376923658 MAE: 0.8522460917908555 Hidden channels: 128 threshold: 10
    [1.0233462811819267, 1.0079267677322608, 0.9442667682212257, 0.8684053091236795, 0.8691430622027366]
    [0.8784572412924094, 0.8793690824394628, 0.845469424789265, 0.8264977936643998, 0.8314369167687403]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9733841327927202 MAE: 0.8773293522463618 Hidden channels: 128 threshold: 11
    [1.0326270473821133, 0.9336924550537221, 0.934796652627277, 0.98071924034663, 0.9850852685538576]
    [0.9032476323987523, 0.8600772334805514, 0.8558794227445101, 0.8862200009507736, 0.8812224716572217]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9943073681789956 MAE: 0.8779992878979647 Hidden channels: 128 threshold: 12
    [0.9724353414970431, 0.9509080445613612, 1.0192695668737888, 1.0230967871318652, 1.0058271008309194]
    [0.8963673760069633, 0.8625191493794886, 0.8717013773227505, 0.8860000408867222, 0.8734084958938987]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9807883615290784 MAE: 0.8812189544699697 Hidden channels: 128 threshold: 13
    [1.0407266567243059, 1.0370097029584324, 0.8825258724678952, 0.9897194558443639, 0.9539601196503941]
    [0.918837210259307, 0.9031277636943649, 0.8359060848829287, 0.8755842685018522, 0.8726394450113955]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.965779442856493 MAE: 0.8627411155845991 Hidden channels: 128 threshold: 14
    [0.8295199519777509, 0.9695137841331056, 0.9651729884466401, 1.0783505875886024, 0.9863399021363661]
    [0.8123154567889052, 0.8425399257967251, 0.8665145051322553, 0.8973357628826982, 0.8949999273224114]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9452526440991154 MAE: 0.8588017292141565 Hidden channels: 128 threshold: 15
    [1.0265337877499678, 0.977139017648181, 0.8345264086463906, 0.9275473211278078, 0.9605166853232296]
    [0.8778498425524356, 0.8691697378574115, 0.8222115032171587, 0.8505184561216467, 0.8742591063221292]

    Dataset: ../Datasets/beauty.pkl RMSE: 0.9871080057530224 MAE: 0.8739795323861237 Hidden channels: 128 threshold: 16
    [1.0146028300863017, 0.9891548814520018, 0.8743046516761019, 1.0566647205279212, 1.000812945022785]
    [0.9049241436195654, 0.84900857465065, 0.8252336801072013, 0.8923301051184824, 0.8984011584347191]

    '''

    '''
    RLMRec with random walk on fashion (RMSE:1.26, MAE: 1.00, 5-fold)
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2480019062535617 MAE: 0.9956890631997656 Hidden channels: 16 threshold: 4
    [1.250821168011174, 1.2247053862480684, 1.2483611063698357, 1.240751859811418, 1.2753700108273118]
    [0.9947181351705264, 0.9928526224436034, 0.9974132328995206, 0.9930054989737037, 1.0004558265114736]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2219896939986685 MAE: 0.9805832213801278 Hidden channels: 16 threshold: 5
    [1.2215172190265862, 1.2264035483039235, 1.216555167505733, 1.242645502965207, 1.202827032191894]
    [0.980478827903566, 0.981578337135607, 0.9813129113335456, 0.9855977440114871, 0.9739482865164331]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1882086868726316 MAE: 0.9681081709444049 Hidden channels: 16 threshold: 6
    [1.1981334454477532, 1.2098062238470724, 1.1826754817444145, 1.185529568171027, 1.1648987151528916]
    [0.972665025818898, 0.9759563562685335, 0.9649984423788096, 0.9658939478640823, 0.9610270823917008]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1589974067683084 MAE: 0.956038144381665 Hidden channels: 16 threshold: 7
    [1.1453259639551703, 1.1696724616462242, 1.1711170128047146, 1.1520428678574148, 1.1568287275780178]
    [0.9480754643806961, 0.9622015483902294, 0.9603054401211202, 0.9583234394347646, 0.9512848295815151]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1292081233341584 MAE: 0.9428466322779796 Hidden channels: 16 threshold: 8
    [1.1482724098773347, 1.1060493891065764, 1.137200724714844, 1.1162976977923587, 1.1382203951796792]
    [0.9544912798553451, 0.9322039574952717, 0.9438282607824806, 0.9409852030799644, 0.9427244601768356]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0990753328967056 MAE: 0.9343963791672927 Hidden channels: 16 threshold: 9
    [1.0841445996142443, 1.0943841319224683, 1.1012307815990716, 1.1053301792112837, 1.110286972136461]
    [0.9283042894326634, 0.9277963878312856, 0.9381424583424435, 0.9374457590751741, 0.9402930011548967]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1002229726510797 MAE: 0.9346789487733064 Hidden channels: 16 threshold: 10
    [1.0858060393696531, 1.0789707263573978, 1.1149563627595942, 1.1156229859828113, 1.1057587487859413]
    [0.9318664509495157, 0.9301890895376776, 0.9435414595552312, 0.9351043879131211, 0.9326933559109862]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0778714015216073 MAE: 0.9220040430711341 Hidden channels: 16 threshold: 11
    [1.1228277538771272, 1.1033532969896847, 1.0424900362608593, 1.0251949501582984, 1.0954909703220672]
    [0.9315731449580572, 0.9388465175184367, 0.9053090002028809, 0.9025670476052, 0.9317245050710954]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.069786592312123 MAE: 0.9190707912649397 Hidden channels: 16 threshold: 12
    [1.1057814312636374, 1.0751291971428667, 1.0821349512561227, 1.0799826443987213, 1.0059047374992673]
    [0.927371449623248, 0.9155576532012982, 0.9320446198336204, 0.921782979886533, 0.8985972537799987
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.098853169741409 MAE: 0.9292845175406323 Hidden channels: 16 threshold: 13
    [1.0679224276924186, 1.0720585050787936, 1.109215048307223, 1.176902324026818, 1.068167543601792]
    [0.9187799062492739, 0.9172356126362976, 0.9264492097928577, 0.9618021996537417, 0.9221556593709913]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0770857084943493 MAE: 0.9177942264204988 Hidden channels: 16 threshold: 14
    [1.0796518611016372, 1.048473393014663, 1.084760572645858, 1.1101685502169898, 1.0623741654925976]
    [0.9243024308767057, 0.9116337213559886, 0.9201531479257511, 0.9199086662264293, 0.9129731657176192]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1016162111391101 MAE: 0.9293368721879599 Hidden channels: 16 threshold: 15
    [1.1580464275240718, 1.0741020095073315, 1.0655785810472176, 1.0766286277937036, 1.1337254098232252]
    [0.9514370764673522, 0.9033702123378917, 0.9207476868021474, 0.9287514668753403, 0.9423779184570684]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.124691129266235 MAE: 0.9335242769080064 Hidden channels: 16 threshold: 16
    [1.1431557811877178, 1.0619081362134095, 1.1319894805875101, 1.202655996064392, 1.0837462522781451]
    [0.9405311800425122, 0.9238286083072804, 0.9360671568958332, 0.9428294542277232, 0.9243649850666833]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2662328128762388 MAE: 0.995435939961934 Hidden channels: 32 threshold: 4
    [1.2704437799308919, 1.2633610984910026, 1.270102233642755, 1.2485242384750537, 1.2787327138414915]
    [0.9944573023054939, 0.9928315263123372, 0.9971950101228898, 0.9928927101693474, 0.9998031508996025]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2278916751912243 MAE: 0.9800604802158359 Hidden channels: 32 threshold: 5
    [1.22512011077618, 1.2285388946410971, 1.227781162071542, 1.2456075185071622, 1.2124106899601401]
    [0.978911757887252, 0.9810221325497734, 0.9811907271183454, 0.9854872809370078, 0.9736905025868007]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1987884450951196 MAE: 0.9684838762201252 Hidden channels: 32 threshold: 6
    [1.2140336391925441, 1.2115901009200265, 1.195836753136553, 1.194522242920624, 1.177959489305851]
    [0.9741809230741248, 0.9754078795502815, 0.966063989428581, 0.9654680563982287, 0.9612985326494105]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1634620334730703 MAE: 0.9567859867710393 Hidden channels: 32 threshold: 7
    [1.1466203544624463, 1.1776467873868324, 1.1690930993755089, 1.1603489825731128, 1.1636009435674513]
    [0.9463086339997578, 0.963584082552722, 0.9615845550137027, 0.9597903550150529, 0.9526623072739607]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.134513368396008 MAE: 0.9424016736850082 Hidden channels: 32 threshold: 8
    [1.1505462030402611, 1.1163015409464105, 1.140229528213844, 1.1182629671908306, 1.1472266025886946]
    [0.9542808528990435, 0.9321579634423176, 0.9412420627112718, 0.9415483728538387, 0.9427791165185687]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1025353627800052 MAE: 0.932810816747477 Hidden channels: 32 threshold: 9
    [1.0916984848999436, 1.099114113407133, 1.1042283829609303, 1.1064693229610052, 1.1111665096710137]
    [0.9285427184922858, 0.928323932467276, 0.9302197151678944, 0.9378965082180757, 0.9390712093918537]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1055579728228908 MAE: 0.9342138352015116 Hidden channels: 32 threshold: 10
    [1.090553961947631, 1.0874756601055318, 1.1186755930868586, 1.1218935342593637, 1.1091911147150688]
    [0.9325117433447235, 0.9355078590880184, 0.9429048049668117, 0.9324152659439053, 0.9277295026640987]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.084229700144435 MAE: 0.9220956599063319 Hidden channels: 32 threshold: 11
    [1.1257900058264692, 1.1043450175956753, 1.0560055615586812, 1.0326031812692211, 1.1024047344721277]
    [0.930314479783963, 0.9378866187325526, 0.9070171010018014, 0.9041605800587217, 0.9310995199546206]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0779581245080805 MAE: 0.9182122355444491 Hidden channels: 32 threshold: 12
    [1.1066372676597895, 1.0817041075972909, 1.0789088421772182, 1.1076323349480945, 1.0149080701580082]
    [0.9249200283740653, 0.9141345110362502, 0.9291277501657914, 0.9240490648385812, 0.8988298233075573]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1079525625970115 MAE: 0.9274148693018232 Hidden channels: 32 threshold: 13
    [1.071743459255574, 1.0950335042062902, 1.1215422805801631, 1.1874515333906277, 1.0639920355524024]
    [0.9198969903683766, 0.9148364254913176, 0.9300522496120637, 0.9600249691934137, 0.9122637118439437]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.0804145320964458 MAE: 0.9179183717895715 Hidden channels: 32 threshold: 14
    [1.0804918544100248, 1.0525928518598255, 1.0689848419613799, 1.1283580490376104, 1.071645063213388]
    [0.9245335219386106, 0.9108801476821649, 0.9138468543904463, 0.9240770869207631, 0.9162542480158722]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1113279461795689 MAE: 0.9285696505979008 Hidden channels: 32 threshold: 15
    [1.1553638985769852, 1.0760712261314427, 1.1189429893919216, 1.0475861413310465, 1.1586754754664481]
    [0.9501725233144619, 0.8986440526644472, 0.9348486049037943, 0.9111373238742201, 0.9480457482325807]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1341389698637108 MAE: 0.9360782241983262 Hidden channels: 32 threshold: 16
    [1.1290051877204441, 1.066027886617468, 1.1571208661139576, 1.2161750998364043, 1.1023658090302788]
    [0.932902339656897, 0.9281334174391102, 0.94731551842993, 0.9451207514536916, 0.9269190940120028]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2706831455992633 MAE: 0.9950596530503978 Hidden channels: 64 threshold: 4
    [1.2708269484058694, 1.2646368698252346, 1.2726451541799408, 1.26553256830422, 1.2797741872810515]
    [0.9941016282360726, 0.9918755434656052, 0.9964212811596511, 0.993565537346846, 0.9993342750438141]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.2294295273502054 MAE: 0.9788371151849983 Hidden channels: 64 threshold: 5
    [1.2268256846447396, 1.2295178495338095, 1.229216508611098, 1.2480128946434978, 1.2135746993178813]
    [0.9773582247578975, 0.9802263929235818, 0.979853457417584, 0.9838945604095718, 0.972852940416356]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.2044102439082518 MAE: 0.9682641330186424 Hidden channels: 64 threshold: 6
    [1.2197002384256401, 1.222432312997026, 1.201279482315639, 1.1984900365927162, 1.180149149210238]
    [0.9745305597677493, 0.9760870297781342, 0.9659969931357263, 0.9652121471883983, 0.9594939352232043]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1706707845633633 MAE: 0.9556217573136809 Hidden channels: 64 threshold: 7
    [1.1560768483923982, 1.1854178413949725, 1.1764943183129108, 1.1658514318242093, 1.1695134828923255]
    [0.9464121479347554, 0.9614577323483123, 0.9579510084251527, 0.9584798321588766, 0.9538080657013075]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1409478827196955 MAE: 0.943461128430814 Hidden channels: 64 threshold: 8
    [1.1597646342208552, 1.1226722525701762, 1.1493167331386236, 1.1225093826453414, 1.150476411023481]
    [0.9573136879536548, 0.9338858361019743, 0.9403019074534157, 0.9416410677834761, 0.9441631428615493]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1062506583499723 MAE: 0.9333671872925761 Hidden channels: 64 threshold: 9
    [1.0935463037289332, 1.1131630254341895, 1.1058851758274735, 1.1067305556129567, 1.1119282311463086]
    [0.9297413939498319, 0.9274875406652474, 0.9310563290741678, 0.9375949950855258, 0.9409556776881075]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1103963432350343 MAE: 0.9353872304711304 Hidden channels: 64 threshold: 10
    [1.0999049458016166, 1.0906255388827064, 1.1271671334967828, 1.1264966203666706, 1.1077874776273968]
    [0.9341064612568578, 0.9349194832579154, 0.9466773002201732, 0.9334589558191203, 0.9277739518015855]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.0950535778021873 MAE: 0.9222063707786348 Hidden channels: 64 threshold: 11
    [1.132919900550125, 1.1195050626099385, 1.055550323327338, 1.0600641799354469, 1.107228422588088]
    [0.9286066059017589, 0.9344312585386584, 0.9045952424561264, 0.9125169937439065, 0.9308817532527236]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.0848797449881442 MAE: 0.9163891916001182 Hidden channels: 64 threshold: 12
    [1.1123828127257607, 1.0909106628836451, 1.085662849264946, 1.1087809217318805, 1.0266614783344896]
    [0.9244162261312131, 0.9158135465720656, 0.9257588611249417, 0.9240695677962483, 0.8918877563761222]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1256996294262032 MAE: 0.9300436562068075 Hidden channels: 64 threshold: 13
    [1.1015617217872278, 1.117295708247753, 1.1197696087033606, 1.2063281966358164, 1.083542911756858]
    [0.9226049160412769, 0.9190458762275878, 0.9238741861867841, 0.9651624055956255, 0.9195308969827627]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.0941870371622135 MAE: 0.9178173650314679 Hidden channels: 64 threshold: 14
    [1.089570966982826, 1.0687741749364168, 1.0667116855056265, 1.1477906341311244, 1.098087724255074]
    [0.9270945581512966, 0.9119122361145743, 0.903848258927354, 0.9269742759183932, 0.9192574960457219]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1335084109202782 MAE: 0.9367837483048579 Hidden channels: 64 threshold: 15
    [1.1721345242206658, 1.136467092103884, 1.117802947875979, 1.0516866116926582, 1.1894508787082043]
    [0.9605894082943054, 0.9173547660893446, 0.9348780055802343, 0.907479778372624, 0.9636167831877813]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1757144428697364 MAE: 0.9492455102616699 Hidden channels: 64 threshold: 16
    [1.1927161997677318, 1.11036071500152, 1.1619789084073755, 1.264028095732897, 1.1494882954391574]
    [0.9571293751610573, 0.9379017179217157, 0.9465265619090361, 0.9622389032957487, 0.9424309930207914]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.270474562551911 MAE: 0.9942969231934555 Hidden channels: 128 threshold: 4
    [1.2695316555068605, 1.2652389572579106, 1.2721343711879771, 1.2658728217479809, 1.279595007058825]
    [0.9920333506082785, 0.9918073480484908, 0.9953292358794658, 0.9934355152745191, 0.9988791661565234]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.2293154521843594 MAE: 0.9785729392612783 Hidden channels: 128 threshold: 5
    [1.2269205920848791, 1.2290571336652898, 1.2289000949522149, 1.2479095112281826, 1.2137899289912295]
    [0.9772520788671614, 0.9796327792917875, 0.9797284526710077, 0.9838170751709738, 0.9724343103054616]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.2054195905530627 MAE: 0.9675629577876329 Hidden channels: 128 threshold: 6
    [1.2211450982594168, 1.2230786990830111, 1.20409502422431, 1.1986821971191086, 1.1800969340794667]
    [0.9715940178861442, 0.9750074791689471, 0.9667418742847662, 0.9653794283591007, 0.9590919892392061]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1734409929277747 MAE: 0.9560711156061916 Hidden channels: 128 threshold: 7
    [1.1566300061758843, 1.1918923932412302, 1.1786422515878392, 1.1673869062193285, 1.1726534074145911]
    [0.9482081337573249, 0.9653038864507955, 0.9563900319985292, 0.9564960580670058, 0.9539574677573032]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1454172375134666 MAE: 0.9436864293405283 Hidden channels: 128 threshold: 8
    [1.1632445484741591, 1.1290706833024025, 1.1513619568583988, 1.1273444182576555, 1.156064580674717]
    [0.9577793185673181, 0.9338424793483714, 0.9399902435179391, 0.9435863742971667, 0.9432337309718457]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1133296799388221 MAE: 0.9309515219414843 Hidden channels: 128 threshold: 9
    [1.1049040598135362, 1.1195068908208476, 1.1122779788403452, 1.1137653315508864, 1.1161941386684953]
    [0.9284332215602035, 0.9276981509010969, 0.9274416142115025, 0.9322988971022861, 0.9388857259323321]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1171311958641594 MAE: 0.935275157607396 Hidden channels: 128 threshold: 10
    [1.1085118313799411, 1.096495209420622, 1.1303541555521792, 1.136102475687864, 1.1141923072801905]
    [0.9332488962212665, 0.9360221334395398, 0.9469699068346422, 0.9337922626802796, 0.9263425888612516]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1019577995801886 MAE: 0.9211430081686964 Hidden channels: 128 threshold: 11
    [1.1434532178629917, 1.1185915998480966, 1.0649608875043186, 1.0668255964902205, 1.1159576961953148]
    [0.9279121647962385, 0.9292629609090486, 0.9048624776447836, 0.9122212189350581, 0.9314562185583533]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.0993602338287345 MAE: 0.9199762828785255 Hidden channels: 128 threshold: 12
    [1.1292942985502599, 1.1080080373873191, 1.0845920661723365, 1.1398411569153186, 1.0350656101184377]
    [0.9271533249283639, 0.9192739032392274, 0.9222379926279182, 0.933985401981358, 0.8972307916157602]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.1354211588903642 MAE: 0.9308708308508538 Hidden channels: 128 threshold: 13
    [1.1013165010215025, 1.1271781870576243, 1.1402132322532859, 1.2183520910423964, 1.0900457830770118]
    [0.9228200001056707, 0.9195934932517952, 0.9290811935147624, 0.9669860582309776, 0.9158734091510626]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.11326157899326 MAE: 0.9251182306436307 Hidden channels: 128 threshold: 14
    [1.1145710751810776, 1.081645545590646, 1.0879417769001871, 1.1623086826516176, 1.1198408146427714]
    [0.9330827334238367, 0.9195814384827143, 0.9153472882656137, 0.9330181159351703, 0.9245615771108183]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.166030024350804 MAE: 0.948328582329407 Hidden channels: 128 threshold: 15
    [1.204282311494758, 1.1545984925281245, 1.1742579670998352, 1.085463076288376, 1.2115482743429258]
    [0.9737427737592733, 0.9312482589905894, 0.9488055827773632, 0.9183840546878671, 0.9694622414319417]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.187715583769996 MAE: 0.9551762990657956 Hidden channels: 128 threshold: 16
    [1.2249627073635725, 1.138263170415818, 1.1514955357277072, 1.2636202966547776, 1.1602362086881042]
    [0.9784298998475888, 0.9505936316280509, 0.9465373776675207, 0.953716773873878, 0.9466038123119398]
    '''
