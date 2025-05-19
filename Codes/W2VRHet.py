import gc
import sys
from transformers import (
    BertModel,
    BertTokenizer,
    DistilBertModel,
    DistilBertTokenizer,
    DistilBertTokenizerFast,
)
from sklearn.model_selection import train_test_split
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
from collections import defaultdict
from tqdm import tqdm
import os
# import nltk
# nltk.download('punkt_tab')

warnings.filterwarnings("ignore", category=FutureWarning, message="'DataFrame.swapaxes' is deprecated")

def compute_hetero_sparsity(data: HeteroData, verbose=True):
    sparsity_dict = {}
    total_observed = 0
    total_possible = 0

    for edge_type in data.edge_types:
        edge_index = data[edge_type].edge_index
        src_type, _, dst_type = edge_type
        num_edges = edge_index.size(1)

        # Determine number of source and destination nodes
        num_src = data[src_type].num_nodes if hasattr(data[src_type], 'num_nodes') else data[src_type].x.size(0)
        num_dst = data[dst_type].num_nodes if hasattr(data[dst_type], 'num_nodes') else data[dst_type].x.size(0)

        possible_edges = num_src * num_dst
        sparsity = 1 - (num_edges / possible_edges) if possible_edges > 0 else 1.0

        # Accumulate totals for overall sparsity
        total_observed += num_edges
        total_possible += possible_edges

        sparsity_dict[edge_type] = {
            'num_edges': num_edges,
            'possible_edges': possible_edges,
            'sparsity': sparsity
        }

        # if verbose:
        #     print(f"{edge_type}: {num_edges} / {possible_edges} => sparsity = {sparsity:.4f}")

    # Compute average sparsity
    overall_sparsity = 1 - (total_observed / total_possible) if total_possible > 0 else 1.0
    # if verbose:
    #     print(f"\nOverall sparsity across all edge types: {overall_sparsity:.4f}")

    return sparsity_dict, overall_sparsity



def train_and_evaluate(training_data, testing_data, items_dict):
    data_train = HeteroData()
    data_test = HeteroData()
    print_counter = 20000

    uid_train = {}
    iid_train = {}
    current_uid_train = 0
    current_iid_train = 0
    rate_count_train = len(training_data)
    counter = 0
    print(rate_count_train)

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
        temp = [target['average_rating'], target['rating_number']] + target['title'].tolist()
        item_features_train.append(temp)
        counter += 1

    counter = 0
    # print("Getting item features (testing)")
    for value in iid_test.keys():
        # if counter % print_counter == 0:
        #     print(str(round(counter / len(iid_test.keys()), 2) * 100) + '%')

        target = items_dict[value]
        temp = [target['average_rating'], target['rating_number']] + target['title'].tolist()
        item_features_test.append(temp)
        counter += 1

    # Adding item nodes with features
    data_train['item'].x = torch.tensor(item_features_train, dtype=torch.float).to(device)  # Item features (2D)
    data_test['item'].x = torch.tensor(item_features_test, dtype=torch.float).to(device)  # Item features (2D)

    # region training edges
    rating_edge_from_train, rating_edge_to_train = [], []
    rating_train = []
    verify_buy_from_train, verify_buy_to_train = [], []
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
            review_train.append(get_word2vec_sentence_vector(row['title'] + row['text'], w2v_model, vector_size))

        if row['verified_purchase']:
            verify_buy_from_train.append(uid_train[row['user_id']])
            verify_buy_to_train.append(iid_train[row['parent_asin']])

        counter += 1

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
    sparsity_stats, avg_sparsity = compute_hetero_sparsity(data_train)
    # print(sparsity_stats)
    print('Training graph sparsity:')
    print(avg_sparsity)
    # print('train edge data finished')

    # region testing edges
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
            review_test.append(get_word2vec_sentence_vector(row['title'] + row['text'], w2v_model, vector_size))

        if row['verified_purchase']:
            verify_buy_from_test.append(uid_test[row['user_id']])
            verify_buy_to_test.append(iid_test[row['parent_asin']])

        counter += 1

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

    sparsity_stats, avg_sparsity = compute_hetero_sparsity(data_test)
    # print(sparsity_stats)
    print('Testing graph sparsity:')
    print(avg_sparsity)

    # print('test edge data finished')

    # Building Heterogeneous graph
    class HeteroGNN(torch.nn.Module):
        def __init__(self, num_users, hidden_channels, item_features_dim):
            super(HeteroGNN, self).__init__()
            self.user_embedding = torch.nn.Embedding(num_users, item_features_dim)

            # HeteroConv for word2vec
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

        def forward(self, x_dict, edge_index_dict):
            # Assuming edge_index_dict is correctly formed and passed
            x_dict['user'] = self.user_embedding(x_dict['user'])  # Embed user features
            x_dict = self.conv1(x_dict, edge_index_dict)  # First layer of convolutions
            x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply non-linearity
            # x_dict = self.conv2(x_dict, edge_index_dict)  # Second layer of convolutions
            # x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply non-linearity
            # x_dict = self.conv3(x_dict, edge_index_dict)
            # x_dict = {key: F.relu(x) for key, x in x_dict.items()}  # Apply non-linearity
            return x_dict

    # Assuming data_train and data_test are defined properly with .x, .edge_index, etc.
    # num_users_train = data_train['user'].num_nodes
    # num_users_test = data_test['user'].num_nodes
    # item_features_dim = data_train['item'].x.size(1)
    #
    # # Instantiate the model
    # model = HeteroGNN(num_users_train, hidden_channels, item_features_dim).to(device)
    #
    # # Training process
    # learning_rate = 0.001
    # num_epochs = 900
    # criterion = torch.nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    #
    # model.train()
    # # Training loop
    # for epoch in range(num_epochs):
    #     optimizer.zero_grad()
    #     out_dict = model(
    #         {
    #             'user': torch.arange(num_users_train).to(device),
    #             'item': data_train['item'].x.to(device)
    #         },
    #         data_train.edge_index_dict
    #     )
    #     user_out = out_dict['user'].to(device)
    #     user_indices = data_train['user', 'rates', 'item'].edge_index[0]
    #     predicted_ratings = model.lin(user_out[user_indices]).squeeze()
    #     loss = criterion(predicted_ratings, data_train['user', 'rates', 'item'].edge_attr.squeeze())
    #     loss.backward()
    #     optimizer.step()
    #     if loss.item() < 0.05:
    #         break
    #     # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')
    #
    # # Evaluation
    # model.eval()
    # with torch.no_grad():
    #     out_dict = model(
    #         {
    #             'user': torch.arange(num_users_test).to(device),
    #             'item': data_test['item'].x.to(device)
    #         },
    #         data_test.edge_index_dict
    #     )
    #     user_out = out_dict['user']
    #     user_indices = data_test['user', 'rates', 'item'].edge_index[0]
    #     predicted_ratings = model.lin(user_out[user_indices]).squeeze().tolist()
    #
    # # print(calculate_RMSE(predicted_ratings, testing_data['rating'].tolist()))
    # # print(calculate_MAE(predicted_ratings, testing_data['rating'].tolist()))
    #
    # return predicted_ratings


def calculate_RMSE(predicted_result, true_label):
    if len(predicted_result) != len(true_label):
        return 0

    total_error = 0
    # individual_diff = []
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
    beauty_path = '../../Datasets/beauty.pkl'
    fashion_path = '../../Datasets/fashion.pkl'
    meta_beauty_path = '../../Datasets/meta_beauty.pkl'
    meta_fashion_path = '../../Datasets/meta_fashion.pkl'
    beauty_w2v_path = 'beauty_w2v_model.model'
    fashion_w2v_path = 'fashion_w2v_model.model'
    num_chunks = 5
    num_folds = num_chunks
    PERCENTAGE_SIZE = 1
    BATCH_SIZE = 250
    threshold_for_fashion = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # threshold_for_fashion = [17, 18, 19, 20]
    threshold_for_beauty = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
    # threshold_for_beauty = [11, 12, 13, 14, 15, 16]
    device = 'cuda'
    hidden_channels = 128
    torch.cuda.manual_seed_all(42)  # If you're using GPU
    np.random.seed(42)
    df_path_list = [beauty_path, fashion_path]
    meta_df_path_list = [meta_beauty_path, meta_fashion_path]
    w2vec_path_list = [beauty_w2v_path, fashion_w2v_path]
    threshold_list = [threshold_for_beauty, threshold_for_fashion]
    count = 0

    for df_path, meta_df_path, w2vec_path, threshold in zip(df_path_list, meta_df_path_list, w2vec_path_list, threshold_list):
        if count == 0:
            count += 1
            continue

        RMSE_list = []
        MAE_list = []

        for local_threshold in threshold:
            df = load_dataset(df_path)
            meta_df = load_dataset(meta_df_path)
            w2vec_path = w2vec_path

            # region pre-process
            # remove nan value from rating column
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
            # df = df[0: 500000]
            # endregion

            # region Word2vec
            sentences = meta_df['title'].tolist() + df['text'].tolist()
            tokenized_titles = [word_tokenize(title) for title in sentences]

            # Set parameters and initialize and train the model
            vector_size = 1  # Dimensionality of the word vectors
            window = 5  # Maximum distance between the current and predicted word within a sentence
            min_count = 1  # Ignores all words with total frequency lower than this num
            workers = 4  # Use these many worker threads to train the model

            # train model if model hasn't trained yet
            w2v_model = Word2Vec(tokenized_titles, vector_size=vector_size, window=window, min_count=min_count, workers=workers)

            # region get train, test dataset ready for word2vec
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

                # Word2vec
                items_dict[row['parent_asin']] = {
                    "average_rating": row['average_rating'],
                    "rating_number": row['rating_number'],
                    "title": get_word2vec_sentence_vector(row['title'], w2v_model, vector_size),
                    "store": row['store']
                }

                counter += 1

            stores = list(set(meta_df['store'].tolist()))

            del df, meta_df, item_with_empty_title, removed_parent_asin

            mae_list = []
            rmse_list = []
            i = 0
            # the train-evaluate process uses 5-Fold Cross-Validation
            while i < num_folds:
                # Dynamically concatenate the chunks for training, excluding the one for validation
                train_chunks = []
                for j in range(num_folds - 1):  # Select (num_folds - 1) chunks for training
                    train_chunks.append(chunks[(i + j) % num_folds])

                train_and_evaluate(
                    pd.concat(train_chunks),
                    chunks[(i + num_folds - 1) % num_folds],  # Validation chunk
                    items_dict
                )
                # Concatenate all the selected chunks for training
                # result = train_and_evaluate(
                #     pd.concat(train_chunks),
                #     chunks[(i + num_folds - 1) % num_folds],  # Validation chunk
                #     items_dict
                # )
                #
                # # Calculate RMSE and MAE for the validation chunk
                # rmse = calculate_RMSE(result, chunks[(i + num_folds - 1) % num_folds]['rating'].tolist())
                # mae = calculate_MAE(result, chunks[(i + num_folds - 1) % num_folds]['rating'].tolist())
                #
                # mae_list.append(mae)
                # rmse_list.append(rmse)
                #
                # # Increment the loop counter
                i += 1
                #
                # # Clear memory
                # gc.collect()
                # torch.cuda.empty_cache()

            # print(
            #     'Dataset:', df_path,
            #     'RMSE:', sum(rmse_list)/len(rmse_list),
            #     "MAE:", sum(mae_list)/len(mae_list),
            #     "Hidden channels:", hidden_channels,
            #     'threshold:', local_threshold
            # )
            #
            # RMSE_list.append(round(sum(rmse_list)/len(rmse_list), 4))
            # MAE_list.append(round(sum(mae_list)/len(mae_list), 4))
            #
            # print(rmse_list)
            # print(mae_list)
            #
            # with open('mae.pkl', 'wb') as f:
            #     pickle.dump(mae_list, f)
            # with open('rmse.pkl', 'wb') as f:
            #     pickle.dump(rmse_list, f)
            #
            # gc.collect()
            # torch.cuda.empty_cache()

        temp = [
            ['RMSE'] + RMSE_list,
            ['MAE'] + MAE_list
        ]

        # Create DataFrame
        df = pd.DataFrame(temp)

        if 'beauty' in df_path:
            output_path = f'../Datasets/Word2vec+RandomWalk+HeteroGNN_hidden_channels={hidden_channels}_beauty.xlsx'
        else:
            output_path = f'../Datasets/Word2vec+RandomWalk+HeteroGNN_hidden_channels={hidden_channels}_fashion.xlsx'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_excel(
            output_path,
            index=False, header=False
        )

