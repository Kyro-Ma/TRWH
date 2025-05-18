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
    beauty_path = '../Datasets/beauty.pkl'
    fashion_path = '../Datasets/fashion.pkl'
    meta_beauty_path = '../Datasets/meta_beauty.pkl'
    meta_fashion_path = '../Datasets/meta_fashion.pkl'
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
            # w2v_model.save(w2vec_path)

            # load trained model directly if model has trained
            # w2v_model = Word2Vec.load(w2vec_path)

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
            output_path = f'../Datasets/Word2vec+RandomWalk+HeteroGNN+InteractionFilter_hidden_channels={hidden_channels}_beauty.xlsx'
        else:
            output_path = f'../Datasets/Word2vec+RandomWalk+HeteroGNN+InteractionFilter_hidden_channels={hidden_channels}_fashion.xlsx'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_excel(
            output_path,
            index=False, header=False
        )

    '''
    word2vec on beauty (RMSE:1.44, MAE: 1.06, 5-fold threshold=2)
    [1.3814050512966536, 1.426625986069026, 1.4840542871223517, 1.4791657221193666, 1.4356650507995092]
    [1.0428584471739415, 1.0559224955834519, 1.0825066071064318, 1.0624696022147326, 1.0684604041696322]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.3474816052976315 MAE: 1.0391960789468477 Hidden channels: 16 threshold: 2
    [1.3413413640833385, 1.335216618592415, 1.3580302635534338, 1.3354610302504413, 1.367358750008529]
    [1.0362886631715273, 1.0329196603406612, 1.0536325408148408, 1.0295586263192504, 1.043580904087959]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.264292598240894 MAE: 0.9947279793074901 Hidden channels: 16 threshold: 3
    [1.335417337341154, 1.2065009675833107, 1.322069181635173, 1.2265150981743593, 1.2309604064704733]
    [1.027129686773917, 0.9711532109008294, 1.0126176213321019, 0.981648510094473, 0.9810908674361289]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.211581164966137 MAE: 0.9704334014901222 Hidden channels: 16 threshold: 4
    [1.287150496473675, 1.2499995362384166, 1.1688093806763493, 1.1858028244312486, 1.166143587010995]
    [1.0036561417542682, 0.9873713550962278, 0.9594332869826259, 0.9511392788791235, 0.9505669447383662]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.0950873320250465 MAE: 0.9231035278201996 Hidden channels: 16 threshold: 5
    [1.0652625902563526, 1.1363249587833182, 1.0996934109219285, 1.110226976340246, 1.063928723823387]
    [0.904285957120497, 0.94214263136619, 0.9249691491641868, 0.9332749819148296, 0.910844919535294]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.1001223596809964 MAE: 0.9259398670241612 Hidden channels: 16 threshold: 6
    [1.1250518681108146, 1.1986397245734315, 1.0404254506823536, 1.0693367949268222, 1.0671579601115602]
    [0.928421497504511, 0.9753810626598994, 0.8923137820726299, 0.9109850780517699, 0.9225979148319968]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.2075965332490057 MAE: 0.971777682822642 Hidden channels: 16 threshold: 7
    [1.4283344767452064, 1.5960188396456647, 1.0400196665075438, 0.9641552934559341, 1.009454389890679]
    [1.0672704187120667, 1.123442697595968, 0.9013451254107213, 0.8683297332971704, 0.8985004390972832]
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9702296221746624 MAE: 0.8706248081106004 Hidden channels: 16 threshold: 8
    [0.9931432427146978, 0.9690294866579989, 0.956228459379983, 0.9401336680559924, 0.9926132540646403]
    [0.8777234910329857, 0.8635858556987175, 0.8639079255100496, 0.8647219164054901, 0.883184851905759]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.5746954464059304 MAE: 1.0807355316171325 Hidden channels: 16 threshold: 9
    [1.1769602573888907, 0.9670209038345385, 1.467703752883456, 0.9716066151107072, 3.290185702812058]
    [0.9815078493988124, 0.874811417862211, 1.0431228905824697, 0.8768265327730919, 1.6274089674690781]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.7540396322604384 MAE: 1.1050833817945036 Hidden channels: 16 threshold: 10
    [1.011279590518926, 2.496780389448145, 1.217657384445612, 3.022061544532549, 1.0224192523569597]
    [0.8842358589667181, 1.3161877066316428, 0.9959024611666373, 1.4398968286233544, 0.8891940535841661]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.1418156696733157 MAE: 0.9610765459577213 Hidden channels: 16 threshold: 11
    [0.9661638774706227, 1.1937914576683468, 1.1410564901937144, 1.1161178008452577, 1.2919487221886365]
    [0.8696292027315857, 0.9870399935612295, 0.9744770473252838, 0.9402358856584101, 1.034000600512097]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.169494640800257 MAE: 0.9570912348901951 Hidden channels: 16 threshold: 12
    [1.2797372149343202, 1.3360108871026601, 0.9442186323807458, 1.3495798160643977, 0.9379266535191596]
    [0.9994765234654553, 1.0571826365500758, 0.844078428991093, 1.0160643003811276, 0.8686542850632238]
    Dataset: ../Datasets/beauty.pkl RMSE: 2.3418345125579845 MAE: 1.3247704755736873 Hidden channels: 16 threshold: 13
    [3.603538782682652, 2.5321840620630796, 2.2648594015664316, 0.9204147006665483, 2.3881756158112104]
    [1.6548449781788883, 1.3347329831474604, 1.3543296543178147, 0.8570138929444945, 1.422930869279778]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.6458131863738088 MAE: 1.0599648847801584 Hidden channels: 16 threshold: 14
    [0.9928692765029546, 1.4531033049856605, 0.9378868917910286, 1.0279275181214023, 3.8172789404679976]
    [0.8929452146955464, 1.0858370739613579, 0.8558601478459085, 0.8998585380899488, 1.5653234493080304]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.618610900831903 MAE: 1.0483341932526422 Hidden channels: 16 threshold: 15
    [1.4287473967970503, 1.946701964704713, 1.2329495480923822, 2.1990325464627127, 1.2856230481026565]
    [1.034736542017355, 1.260168492384644, 1.0052530499032744, 0.9477639300418692, 0.9937489519160676]
    Dataset: ../Datasets/beauty.pkl RMSE: 1.6316578470522969 MAE: 1.077901264350179 Hidden channels: 16 threshold: 16
    [0.8772947301971994, 1.1607494446472542, 2.3237957009909667, 1.0555871678408788, 2.740862191585186]
    [0.8412281386339993, 0.9790744796011507, 1.2318013234488447, 0.9012105090285896, 1.4361918710383104]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9694913965076477 MAE: 0.871215953972673 hidden_channels: 32 threshold: 8
    [0.9754153890923051, 0.9762721368227888, 0.9419431344969631, 0.9555303863416593, 0.998295935784522]
    [0.872374014166966, 0.8689253425768947, 0.8655381001179107, 0.864367238287976, 0.8848750747136175]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9496117569310156 MAE: 0.8639206925548649 Hidden channels: 64 threshold: 10
    [0.915448293020342, 0.9797468362181906, 0.8973018185232667, 0.9655539339784004, 0.9900079029148787]
    [0.8598988214032023, 0.8783460357311679, 0.8439511127198724, 0.8719380913844841, 0.8654694015355978]
    '''

    '''
    Dataset: ../Datasets/fashion.pkl RMSE: 1.3912959810244838 MAE: 1.037076529392956 Hidden channels: 16 threshold: 2
    [1.2618654923134796, 1.7684345908505625, 1.300044324329358, 1.300104601164194, 1.3260308964648257]
    [1.0004213067617684, 1.1249431293047558, 1.0199209647313636, 1.0217727264109975, 1.018324519755894]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.280576532334588 MAE: 1.0095237826707895 Hidden channels: 16 threshold: 3
    [1.3011936875679218, 1.2629329976475838, 1.2963567122423671, 1.2534954444803241, 1.2889038197347429]
    [1.0163915086234523, 1.004502525769848, 1.0128430156741963, 0.9981492865095232, 1.0157325767769272]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2512052975668229 MAE: 0.9955684034530725 Hidden channels: 16 threshold: 4
    [1.2576897623884773, 1.2788191790325412, 1.221238819542017, 1.2795900209912345, 1.2186887058798443]
    [0.9963620552500053, 1.0078432550321106, 0.9819706709949229, 1.005518314485241, 0.9861477215030832]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2432746982502088 MAE: 0.988581044789313 Hidden channels: 16 threshold: 5
    [1.3292936212210478, 1.2370811992411652, 1.187631438224737, 1.2298104655770767, 1.232556766987017]
    [1.0232551264020233, 0.9867519962881385, 0.9721415369828025, 0.977950993695398, 0.9828055705782027]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2302134247412295 MAE: 0.989756716547423 Hidden channels: 16 threshold: 6
    [1.200147214018855, 1.2241549006820642, 1.2047425612838478, 1.2545991892518742, 1.267423258469507]
    [0.970482085078862, 0.9849110988469715, 0.9797955265122986, 1.006528767665132, 1.007066104633851]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.4661428013093962 MAE: 1.0671525842456382 Hidden channels: 16 threshold: 7
    [2.4096596557352434, 1.1631362378921306, 1.2086992830863423, 1.2999349481504536, 1.2492838816828113]
    [1.3775899603717054, 0.964452313017677, 0.9795390700575386, 1.0086627448713492, 1.0055188329099207]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2566499488445466 MAE: 0.9948613379173639 Hidden channels: 16 threshold: 8
    [1.215511841243388, 1.1579415264270845, 1.1785959738288223, 1.449995474226661, 1.2812049284967768]
    [0.9724893802157459, 0.9492903371221094, 0.9545792562698806, 1.0962073400955459, 1.0017403758835375]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1504014061706473 MAE: 0.9466745965443637 Hidden channels: 16 threshold: 9
    [1.1400952522286365, 1.118448806895333, 1.1217187195869953, 1.161051632745755, 1.2106926193965182]
    [0.941357015094462, 0.9344413824829432, 0.9406106405679484, 0.9522442380198834, 0.964719706556582]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.3085231583094197 MAE: 1.0314943955060436 Hidden channels: 16 threshold: 10
    [1.6145160038417639, 1.0670767719515515, 1.2127973288315124, 1.1545973529261448, 1.4936283339961263]
    [1.1998803216755562, 0.9135358215635264, 0.9695790226941886, 0.9455020352320902, 1.1289747763648559]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1081222270904922 MAE: 0.9292783931342031 Hidden channels: 16 threshold: 11
    [1.1512779986380597, 1.0964287818204597, 1.1071356605819953, 1.0498689903100666, 1.1358997041018801]
    [0.9418417475152863, 0.9230946377130357, 0.9262333973379417, 0.9158178002256752, 0.9394043828790771]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.246847488577594 MAE: 0.9877439799328165 Hidden channels: 16 threshold: 12
    [1.176923083354551, 1.0383261817247271, 1.1731595420023173, 1.1243050984510166, 1.7215235373553575]
    [0.9639271105343429, 0.8931040348619713, 0.9522443199326935, 0.9364510395310773, 1.192993394803997]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.135521906348244 MAE: 0.9449989610613789 Hidden channels: 16 threshold: 13
    [1.244547892640567, 1.0943313664860088, 1.2496467319377753, 1.03110486651402, 1.0579786741628494]
    [0.9762486988623542, 0.9288910009987502, 1.0137035279086428, 0.9069717038024839, 0.8991798737346642]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2838969051348461 MAE: 1.0021468351706744 Hidden channels: 16 threshold: 14
    [1.8075632850157572, 1.0904100394574454, 1.124919735701424, 1.1513183690786675, 1.2452730964209373]
    [1.2104309251094616, 0.9148809858292464, 0.9597030510356707, 0.9388254404685097, 0.9868937734104841]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2776580268484756 MAE: 1.0002958199851668 Hidden channels: 16 threshold: 15
    [1.121728527591782, 1.0307188163155443, 0.9977549084475925, 1.3192171948730906, 1.918870687014369]
    [0.9228255126982912, 0.8989301043094671, 0.8835788918486086, 1.030837850752512, 1.2653067403169551]
    Dataset: ../Datasets/fashion.pkl RMSE: 1.081365936373632 MAE: 0.916687474854092 Hidden channels: 16 threshold: 16
    [1.0189842199076957, 1.1234723155110267, 1.122392266316146, 1.1014479357218145, 1.040532944411478]
    [0.8936677822464085, 0.9178207207225267, 0.9372031833126467, 0.9315689490762303, 0.9031767389126482]

    Dataset: ../Datasets/fashion.pkl RMSE: 1.127185515814879 MAE: 0.9336670439985255 hidden_channels: 32 threshold: 10
    [1.1648791024237317, 1.1214704316089406, 1.1040414653872768, 1.1702994127171935, 1.0752371669372525]
    [0.9506096219332726, 0.9285648602577053, 0.9257006719010438, 0.9467244621513113, 0.9167356037492946]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0787749910808042 MAE: 0.9124508562432112 hidden_channels: 32 threshold: 14
    [1.1046963361206275, 1.092892962078783, 1.0670481385455617, 1.1397595787454013, 0.989477939913647]
    [0.9129952573762198, 0.9144799530136376, 0.9173566162063136, 0.9378875760840045, 0.8795348785358806]
        
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0736764093234714 MAE: 0.9089314661646185 Hidden channels: 64 threshold: 15
    [1.0739425874972066, 1.0362367744045196, 0.9991764445376865, 1.1269491883355087, 1.1320770518424361]
    [0.9045801868576304, 0.894879019959913, 0.882169772076457, 0.91739543237505, 0.9456329195540425]
    '''

    '''
    Dataset: ../Datasets/beauty.pkl RMSE: 1.4804714494834528 MAE: 1.0844082020959342 Hidden channels: 32 threshold: 2
    [1.5150597568782018, 1.5941764461794323, 1.3686198588191367, 1.5816065953837135, 1.34289459015678]
    [1.1068613792941937, 1.1209792068867752, 1.0479579285266585, 1.1142278578454405, 1.0320146379266035]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.2701091892375242 MAE: 0.9936206752476343 Hidden channels: 32 threshold: 3
    [1.2404249259754376, 1.2665337902000922, 1.2466030419124288, 1.2884893760760847, 1.3084948120235789]
    [0.981121947693056, 0.9920369532785303, 0.9836606468948692, 1.0069263418869803, 1.0043574864847356]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.1936668535574682 MAE: 0.9591423377490589 Hidden channels: 32 threshold: 4
    [1.1978336047238267, 1.1839812898095106, 1.220212552355599, 1.2038201220538098, 1.162486698844595]
    [0.9559949165687464, 0.9582455855539442, 0.9824364314464445, 0.953494252657556, 0.9455405025186029]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.117540478458704 MAE: 0.9308228680426758 Hidden channels: 32 threshold: 5
    [1.076844897412091, 1.111599734480102, 1.086488792611451, 1.208346587234974, 1.1044223805549007]
    [0.9194645709067785, 0.9212032959816919, 0.9159046611592284, 0.9688414919190258, 0.928700320246655]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.0540510041511062 MAE: 0.9029258339519715 Hidden channels: 32 threshold: 6
    [1.1006527988343184, 1.0482231048272865, 1.062290888085167, 1.037558382168765, 1.0215298468399947]
    [0.9195300022096361, 0.898007836679341, 0.8981180928099616, 0.9012353881119743, 0.8977378499489435]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.999625583730141 MAE: 0.8859402791973702 Hidden channels: 32 threshold: 7
    [1.00349847305839, 1.0132061857719161, 1.0368621263241438, 0.9535596626160527, 0.9910014708802025]
    [0.892601719279784, 0.8788823981564777, 0.8983862778652734, 0.8771309162694122, 0.8827000844159039]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9622728942683931 MAE: 0.868522519168868 Hidden channels: 32 threshold: 8
    [0.9676375994494566, 0.980940790881825, 0.9555526253200308, 0.9268576342114415, 0.9803758214792119]
    [0.8728939280643055, 0.8629984121811373, 0.8740147196384768, 0.8560600513947004, 0.8766454845657201]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9915267128014044 MAE: 0.8845748252541232 Hidden channels: 32 threshold: 9
    [0.9739947165799125, 1.0579619740354205, 0.998682387681977, 0.9624091147272602, 0.9645853709824516]
    [0.8781129494915119, 0.9200703191982396, 0.8900047001419266, 0.8685260093228984, 0.86616014811604]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.0198653988060016 MAE: 0.8928637118202861 Hidden channels: 32 threshold: 10
    [1.2132557008692963, 0.9712573664314358, 0.9154176962467693, 0.9932147035845021, 1.0061815268980048]
    [0.9816716477352884, 0.866645882538381, 0.8547757279942134, 0.8788795899804608, 0.882345710853087]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9460432798692757 MAE: 0.8658606602171555 Hidden channels: 32 threshold: 11
    [0.92746679266, 0.8995453728435956, 0.9566147810739869, 0.9838401834261331, 0.9627492693426634]
    [0.8588499280861596, 0.8509466869247945, 0.870376894762942, 0.8782812015044115, 0.8708485898074688]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.1424125568420356 MAE: 0.93270089695096 Hidden channels: 32 threshold: 12
    [1.2098541868525805, 0.9535135411628861, 0.9362885595339382, 0.9708704159281938, 1.6415360807325792]
    [0.9859733077132735, 0.8686885944420792, 0.8489346899350763, 0.8693744001593335, 1.0905334925050374]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9681034476538729 MAE: 0.8673950742829388 Hidden channels: 32 threshold: 13
    [1.0367531443820603, 0.9538086014105617, 0.9102457744495132, 0.9550428947805005, 0.9846668232467287]
    [0.9074259014077725, 0.855053316855943, 0.8442624048007543, 0.8642007014049055, 0.866033046945319]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.096665201798831 MAE: 0.9155307570703419 Hidden channels: 32 threshold: 14
    [1.5324836523698249, 1.1175875308980276, 0.9590674392366915, 0.8948282761894835, 0.9793591103001268]
    [1.0909062949122428, 0.9162337945063267, 0.8661710299192601, 0.8432429797235762, 0.861099686290304]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.3201294287581322 MAE: 0.9445805608842777 Hidden channels: 32 threshold: 15
    [0.978437709429432, 0.9185331469586715, 1.0968793415289495, 2.5990299237074046, 1.0077670221662038]
    [0.8907353540564906, 0.8514859975802106, 0.9276910582322383, 1.1562439675179352, 0.896746427034514]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.3928165982871366 MAE: 0.9708892226020034 Hidden channels: 32 threshold: 16
    [0.9853752928859392, 1.0613092937744755, 2.937404650533721, 0.9984926763657078, 0.9815010778758386]
    [0.8910302813141828, 0.9071210801937575, 1.3085862196770468, 0.8812907078392525, 0.8664178239857773]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.403287415598332 MAE: 1.051169059839054 Hidden channels: 32 threshold: 2
    [1.3347510659717048, 1.3148644512955459, 1.3633725956537752, 1.4972225013172002, 1.5062264637534348]
    [1.0251952979389651, 1.0181507546089992, 1.0404541887878827, 1.0805851906974628, 1.0914598671619595]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2858421243361196 MAE: 1.012977663592578 Hidden channels: 32 threshold: 3
    [1.3668370530899137, 1.269393370915159, 1.2573535612684272, 1.2938427188558106, 1.241783917551288]
    [1.0440617938746415, 1.0064108189723813, 1.0041822925669497, 1.012169462340483, 0.998063950208434]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2695073076479726 MAE: 1.0012150303640222 Hidden channels: 32 threshold: 4
    [1.2561866300768996, 1.2427235939475492, 1.2309430506194576, 1.2598907364259035, 1.3577925271700537]
    [0.9988390604608675, 0.9889635225827825, 0.9860456191458951, 0.9962752724575477, 1.0359516771730173]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2611687839201422 MAE: 0.9965423397213646 Hidden channels: 32 threshold: 5
    [1.2253564638229333, 1.2609953116005086, 1.2196869698695514, 1.3139885066018044, 1.285816667705913]
    [0.9777750615669369, 0.9983050899227547, 0.9804064437988025, 1.0145131814612147, 1.0117119218571142]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1966752062155936 MAE: 0.9717901810086971 Hidden channels: 32 threshold: 6
    [1.1783115488478622, 1.2049322467567036, 1.1775319147418277, 1.2363513901298817, 1.186248930601694]
    [0.9624704537277029, 0.9750624281162596, 0.9608313271111202, 0.99022097040305, 0.9703657256853533]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1814437192293863 MAE: 0.9606845509956242 Hidden channels: 32 threshold: 7
    [1.1747516156550624, 1.1733504837698898, 1.2084409043245545, 1.1731725071372001, 1.1775030852602242]
    [0.9548304818496525, 0.9643533730485689, 0.9687604245302888, 0.9594672152870364, 0.9560112602625745]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1743074396239463 MAE: 0.9574161078761649 Hidden channels: 32 threshold: 8
    [1.2625720612587619, 1.1690884186575687, 1.1488027817655362, 1.1510576240848243, 1.140016312353041]
    [0.9922475767346745, 0.9547134272657737, 0.9434525562188277, 0.950193307407352, 0.946473671754197]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1500212521188569 MAE: 0.9467921472847302 Hidden channels: 32 threshold: 9
    [1.1642418006935458, 1.150872414638115, 1.1504608870769644, 1.1360782554262412, 1.148452902759418]
    [0.9517232236362431, 0.9461687726087437, 0.9495324705624777, 0.9407384062071997, 0.9457978634089863]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1296579051520463 MAE: 0.9373936002950412 Hidden channels: 32 threshold: 10
    [1.1489457854036813, 1.115146431069338, 1.1463287530794157, 1.154615693149778, 1.083252863058019]
    [0.9492427143980401, 0.9343821287603853, 0.940928901843351, 0.94461508049316, 0.917799175980269]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0961893576343908 MAE: 0.9173324537319081 Hidden channels: 32 threshold: 11
    [1.1157806695324886, 1.076307419049083, 1.0969634611312469, 1.0948864718505926, 1.0970087666085435]
    [0.9244674553447528, 0.9063559662129552, 0.9177799260964932, 0.9223857777950267, 0.9156731432103125]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.103286484638814 MAE: 0.9246899725907143 Hidden channels: 32 threshold: 12
    [1.0903992299499243, 1.1058116456351412, 1.1428882375196108, 1.0902777894232552, 1.087055520666139]
    [0.9215631561058648, 0.9206407634265564, 0.9373623680367442, 0.92374709471497, 0.9201364806694363]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1670282324254422 MAE: 0.9582059835074844 Hidden channels: 32 threshold: 13
    [1.2434578600140993, 1.104683702242481, 1.0823999285613106, 1.0803452979840562, 1.3242543733252652]
    [0.9726072690818568, 0.911036478655394, 0.9187280928738906, 0.9282853961185576, 1.0603726808077232]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.100075369874376 MAE: 0.9187510983520258 Hidden channels: 32 threshold: 14
    [1.1026661594533225, 1.140508404937243, 1.0682293468134423, 1.1345809430339546, 1.0543919951339176]
    [0.9139787364585885, 0.9262248191454906, 0.9145312396491263, 0.9435660412483509, 0.8954546552585723]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0713081378499063 MAE: 0.9109037177638862 Hidden channels: 32 threshold: 15
    [1.0613875599293348, 1.0088929968306832, 1.0496702033525003, 1.1128418396348303, 1.1237480895021827]
    [0.9035948956909714, 0.8916690310255266, 0.9024036972090144, 0.9147456667220137, 0.9421052981719054]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0947990866734716 MAE: 0.914367918278989 Hidden channels: 32 threshold: 16
    [1.0594224315872898, 1.099371407932022, 1.1576984628260816, 1.1177097888794367, 1.0397933421425285]
    [0.9062987805949385, 0.9095003021072594, 0.93920451505359, 0.921911415922103, 0.8949245777170544]
    '''

    '''
    Dataset: ../Datasets/beauty.pkl RMSE: 1.4313192223393674 MAE: 1.0604644785369939 Hidden channels: 64 threshold: 2
    [1.5024495680202334, 1.4050129773669569, 1.484245832908318, 1.405885953477834, 1.3590017799234952]
    [1.0856285558483718, 1.0544233269354373, 1.076908710128679, 1.052021990321062, 1.0333398094514192]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.2693771160215253 MAE: 0.9923831768306514 Hidden channels: 64 threshold: 3
    [1.2600990726620438, 1.2211181225611758, 1.2513511181829478, 1.2573777335361256, 1.3569395331653327]
    [0.9880251432819389, 0.9758557085417073, 0.9843665441602918, 0.9863764766070011, 1.0272920115623172]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.20233616153018 MAE: 0.9610248840477723 Hidden channels: 64 threshold: 4
    [1.175386726774687, 1.2122565559491336, 1.2174736045447836, 1.2354942138306937, 1.1710697065516025]
    [0.9555300985254519, 0.9615636789381731, 0.971056854223798, 0.9678729618667571, 0.9491008266846821]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.1078663070542552 MAE: 0.9250552502342929 Hidden channels: 64 threshold: 5
    [1.0825676819780765, 1.1233159754886741, 1.1572037854569803, 1.0960323953592381, 1.080211696988307]
    [0.9166807065485205, 0.9232101870199606, 0.9460215444215728, 0.9196481332024998, 0.9197156799789112]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.0819276996114826 MAE: 0.91588316786678 Hidden channels: 64 threshold: 6
    [1.1419920470643898, 1.0664773656991149, 1.0983305385095345, 1.0852480770213053, 1.0175904697630689]
    [0.9341485295286772, 0.9069976367752086, 0.9230562096134044, 0.9192256983587035, 0.8959877650579066]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9989621979549886 MAE: 0.8838319163156949 Hidden channels: 64 threshold: 7
    [0.9944245580275061, 1.0024491609862898, 1.027364024325822, 0.9600453389171688, 1.0105279075181566]
    [0.8845131073704188, 0.8750097919564209, 0.891747781188586, 0.8762170525420034, 0.8916718485210455]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9687865251045477 MAE: 0.8681159958337739 Hidden channels: 64 threshold: 8
    [0.9944742756437517, 0.9897196595125495, 0.9569107579083407, 0.9449272291918727, 0.9579007032662232]
    [0.8794894273654911, 0.8642645733324723, 0.8759139336839414, 0.8548431777332857, 0.8660688670536788]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9767705563890923 MAE: 0.8764183039800992 Hidden channels: 64 threshold: 9
    [0.9812099981141889, 0.9657264287309877, 1.0210745431459305, 0.9856941745844474, 0.9301476373699067]
    [0.8838581871323589, 0.8632289640520339, 0.8971760655458043, 0.8758663301681927, 0.8619619730021062]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9527276906074903 MAE: 0.8626082602003627 Hidden channels: 64 threshold: 10
    [0.995593007841921, 0.9024389162676366, 0.944058966901595, 0.9525959210918926, 0.9689516409344063]
    [0.8705279864171517, 0.8459454383078666, 0.8705032008119392, 0.865464152366639, 0.8606005230982166]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.0681161568263422 MAE: 0.9165026225162706 Hidden channels: 64 threshold: 11
    [0.9634778009922762, 0.9378155577451099, 0.951906644739165, 1.5449302736691708, 0.9424505069859889]
    [0.8712295380459807, 0.8621268356579868, 0.8625368251182224, 1.1241490047154923, 0.8624709090436707]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.0240472353667167 MAE: 0.8740081162281133 Hidden channels: 64 threshold: 12
    [0.9496196451808356, 0.9678671742147049, 0.9520299469779058, 0.9786709494361276, 1.2720484610240093]
    [0.8736893434636953, 0.8741762377176864, 0.8487073855168992, 0.8904737051911791, 0.8829939092511067]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9326921900569054 MAE: 0.8496196501854062 Hidden channels: 64 threshold: 13
    [0.9192352775066513, 0.9682338006819982, 0.9006631693125999, 0.8940100849444668, 0.981318617838811]
    [0.8519071298647981, 0.8491788402178116, 0.8380714331212625, 0.8445099040840992, 0.8644309436390595]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 0.9483998176801594 MAE: 0.8622476826376758 Hidden channels: 64 threshold: 14
    [1.0020139808191022, 0.9990755363980319, 0.9576432894937799, 0.8790180009891642, 0.9042482807007183]
    [0.887283139991958, 0.8811035542579393, 0.8643927643943551, 0.8363625668632665, 0.8420963876808605]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.333587551727628 MAE: 0.9152950712908934 Hidden channels: 64 threshold: 15
    [1.0776247704589796, 1.0260746812753492, 1.1026903972637525, 2.3734180845060773, 1.0881298251339826]
    [0.9098761133294327, 0.870464558099872, 0.9365051505721264, 0.9754324446659514, 0.8841970897870848]
    
    Dataset: ../Datasets/beauty.pkl RMSE: 1.2497129470862987 MAE: 0.9743685725563787 Hidden channels: 64 threshold: 16
    [2.081042450674861, 1.0926768471263038, 1.0596259361279374, 0.9857337846925778, 1.0294857168098142]
    [1.3277364907172224, 0.8998986283969197, 0.8991562855577462, 0.8605473138406787, 0.8845041442693264]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.333333750558792 MAE: 1.0281287733253224 Hidden channels: 64 threshold: 2
    [1.3778092633328434, 1.281326990928186, 1.3613285026376107, 1.3237346415841411, 1.3224693543111792]
    [1.038282912191635, 1.0079191556806844, 1.0407575687188462, 1.0291003170368227, 1.024583912998623]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2920289107259797 MAE: 1.0062512147022438 Hidden channels: 64 threshold: 3
    [1.259613735216777, 1.2522120329795339, 1.2325348148467181, 1.4784815060333913, 1.2373024645534771]
    [0.9980634437283346, 0.9930775005453456, 0.9904872813491035, 1.0589953104662608, 0.9906325374221754]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.29455771493104 MAE: 1.0131899190215763 Hidden channels: 64 threshold: 4
    [1.399763965752372, 1.2626105619366008, 1.338042885176232, 1.247853150984205, 1.2245180108057907]
    [1.0537823528372403, 1.002967683707201, 1.030663310214512, 0.9921075154192156, 0.9864287329297116]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.2854183261255696 MAE: 1.0042043255692235 Hidden channels: 64 threshold: 5
    [1.3222678613137613, 1.3179894214087224, 1.1944308033528073, 1.364033056118234, 1.2283704884343218]
    [1.0096688155035973, 1.0252131083968739, 0.9708179083592453, 1.0340860667957157, 0.9812357287906852]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.253343525825037 MAE: 0.9908908607061226 Hidden channels: 64 threshold: 6
    [1.4078436365181943, 1.191218629391269, 1.2196204459408369, 1.2069057268578862, 1.2411291904169979]
    [1.048263412143643, 0.97090762130257, 0.977642918964192, 0.9734703314452181, 0.9841700196749895]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1957390474287917 MAE: 0.9634639098623516 Hidden channels: 64 threshold: 7
    [1.1788125808971086, 1.1878761865451206, 1.2292005041724792, 1.1818943649550449, 1.200911600574205]
    [0.9563064157979766, 0.9685284083618124, 0.9751212960428811, 0.9573145008700893, 0.9600489282389989]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.164580326671619 MAE: 0.9531228929705616 Hidden channels: 64 threshold: 8
    [1.2091081827709527, 1.2039036973037525, 1.154418622859643, 1.087745124259388, 1.1677260061643584]
    [0.9721466695366938, 0.9585959841307379, 0.9535162124087476, 0.9278305333163693, 0.9535250654602597]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1393469281311552 MAE: 0.9433462881528023 Hidden channels: 64 threshold: 9
    [1.1327377584759744, 1.140710049608375, 1.1330717727536679, 1.165554179664865, 1.1246608801528946]
    [0.9385211451658311, 0.9453666884819241, 0.9456874012278899, 0.9588269504866456, 0.9283292554017208]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1419798966677805 MAE: 0.9413695023361619 Hidden channels: 64 threshold: 10
    [1.2739406869109182, 1.077094788323565, 1.1570999311870926, 1.1352795494120744, 1.0664845275052517]
    [0.9945779336869832, 0.920770071222669, 0.9417130497411195, 0.9382716694304984, 0.9115147875995395]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.092461380227265 MAE: 0.9172555893409984 Hidden channels: 64 threshold: 11
    [1.1404213100672635, 1.0755630278875536, 1.0881259283429159, 1.0568073191914265, 1.1013893156471655]
    [0.9330900009476438, 0.9085944161930519, 0.9131252019129291, 0.913887117414179, 0.9175812102371877]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1260071088495978 MAE: 0.9323931167947487 Hidden channels: 64 threshold: 12
    [1.1459188190978093, 1.0478269482836355, 1.1467124576530183, 1.1584960221258707, 1.131081297087656]
    [0.9374753310277919, 0.9005306836626868, 0.9347874832223203, 0.9538054240307724, 0.9353666620301724]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.1162776196727944 MAE: 0.9258811521349498 Hidden channels: 64 threshold: 13
    [1.213866265076086, 1.1226360201698231, 1.0560662480114344, 1.0722085722604378, 1.116610992846191]
    [0.9620717314350097, 0.9122391652556054, 0.9105980591491899, 0.9249948993386244, 0.9195019054963199]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0822908977449754 MAE: 0.9095566936046258 Hidden channels: 64 threshold: 14
    [1.0956646259296257, 1.0811469750698683, 1.0714716706254883, 1.154154523296535, 1.0090166938033607]
    [0.9054053091695534, 0.9003493131468187, 0.919177763001144, 0.9375434355928907, 0.8853076471127221]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0964410382808762 MAE: 0.9220280840987416 Hidden channels: 64 threshold: 15
    [1.0896308646936208, 1.0268821229873104, 0.9793375744897541, 1.1701254906945937, 1.2162291385391024]
    [0.9132989680312669, 0.898501672789898, 0.8628572900425805, 0.9454230585571906, 0.9900594310727716]
    
    Dataset: ../Datasets/fashion.pkl RMSE: 1.0958685052234647 MAE: 0.9155335958387232 Hidden channels: 64 threshold: 16
    [1.0216028509660873, 1.1046790400596818, 1.1279241570345695, 1.1311824952502263, 1.093953982806758]
    [0.889590754899448, 0.9142868941014607, 0.9305504231291956, 0.9249950904799498, 0.9182448165835618]
    '''

    '''
    Dataset: ../Datasets/beauty.pkl RMSE: 1.4330358607384817 MAE: 1.061394078864615 Hidden channels: 128 threshold: 2
[1.3990397243469206, 1.4487675166581884, 1.3963382419272758, 1.4330595241257638, 1.48797429663426]
[1.0496787212619318, 1.058968727227738, 1.0488096363129875, 1.0673045210446033, 1.0822087884758147]
Dataset: ../Datasets/beauty.pkl RMSE: 1.3701423987311432 MAE: 1.0254093760076164 Hidden channels: 128 threshold: 3
[1.307272918311214, 1.3300279359149039, 1.4532604500509563, 1.4380724782742247, 1.3220782111044178]
[1.0078694680574691, 1.0200564634313631, 1.0556431009979836, 1.0351596157231822, 1.0083182318280828]
Dataset: ../Datasets/beauty.pkl RMSE: 1.2339963216739114 MAE: 0.976526089929251 Hidden channels: 128 threshold: 4
[1.187806656193965, 1.2300795392697021, 1.1619583598180179, 1.3729187200675845, 1.2172183330202873]
[0.9622992184452828, 0.9664685393597454, 0.9513992491207013, 1.034881455139633, 0.9675819875808923]
Dataset: ../Datasets/beauty.pkl RMSE: 1.1051772945254537 MAE: 0.9211906252011163 Hidden channels: 128 threshold: 5
[1.087912359607763, 1.1235858401217207, 1.1295989014481178, 1.1078959970599656, 1.0768933743897013]
[0.9125116451803522, 0.9256296183215683, 0.925133559603525, 0.9240134619845827, 0.9186648409155534]
Dataset: ../Datasets/beauty.pkl RMSE: 1.068187114901742 MAE: 0.9085236356447318 Hidden channels: 128 threshold: 6
[1.0942568984950087, 1.0749018876028367, 1.0552309467528627, 1.0625020925607143, 1.0540437490972872]
[0.9162431407274795, 0.9096756537019469, 0.9009385725428252, 0.9049822212225128, 0.9107785900288948]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9981606771248364 MAE: 0.8820198961586204 Hidden channels: 128 threshold: 7
[0.9984397433971283, 1.0418630378623055, 1.0348341200862798, 0.9329674270940864, 0.9826990571843826]
[0.8891841792304603, 0.8901378211264466, 0.8927984591239186, 0.8682290383826864, 0.8697499829295907]
Dataset: ../Datasets/beauty.pkl RMSE: 0.972440000345087 MAE: 0.8691137123177143 Hidden channels: 128 threshold: 8
[0.9682002814226466, 0.9745401992999784, 0.9730188664437086, 0.9525711276553985, 0.9938695269037037]
[0.8661241595938413, 0.8645727007343006, 0.8745860127548563, 0.8563361539092924, 0.8839495345962801]
Dataset: ../Datasets/beauty.pkl RMSE: 0.998449737177048 MAE: 0.8855637027022574 Hidden channels: 128 threshold: 9
[1.0398503045346683, 0.9714862953698896, 1.0000075756976736, 0.9773055835632389, 1.0035989267197691]
[0.9081612895085008, 0.8757946456745982, 0.8915572456987274, 0.8726248511490623, 0.8796804814803983]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9359009154163246 MAE: 0.8580541160494324 Hidden channels: 128 threshold: 10
[0.9306843177507845, 0.9181987647740234, 0.8982899579188317, 0.9563381976657237, 0.9759933389722604]
[0.8609460414866502, 0.844680907416632, 0.8496341224192039, 0.8678345015707329, 0.8671750073539433]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9325718257173978 MAE: 0.8599774426233356 Hidden channels: 128 threshold: 11
[0.9386181495854831, 0.9189220032039961, 0.9193836809149062, 0.9663003136737964, 0.9196349812088074]
[0.8564117687898553, 0.8643986441283581, 0.8547288209334862, 0.8726644729500405, 0.8516835063149378]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9505038563827111 MAE: 0.8657049616904418 Hidden channels: 128 threshold: 12
[0.9644711082244221, 0.9599650936545651, 0.9659072873668688, 0.935217442128712, 0.9269583505389876]
[0.8770619643358107, 0.8736908963804924, 0.8534897026268246, 0.8583063440342938, 0.865975901074788]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9571505975191623 MAE: 0.8635502150393414 Hidden channels: 128 threshold: 13
[0.9790747770292081, 0.9982094652090201, 0.8822783040500066, 0.873290774841412, 1.0528996664661647]
[0.8850533449521315, 0.8783213427803979, 0.8380857980108088, 0.833441662183952, 0.8828489272694168]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9672011721428726 MAE: 0.8600843728903855 Hidden channels: 128 threshold: 14
[0.9898788374456815, 1.0480400437050301, 0.9198426630685709, 0.9547247263435554, 0.9235195901515245]
[0.8679672043338671, 0.8906116835081629, 0.8402786718020976, 0.8585880672043585, 0.8429762376034413]
Dataset: ../Datasets/beauty.pkl RMSE: 1.0392068502249021 MAE: 0.8805287922830969 Hidden channels: 128 threshold: 15
[0.9933557098576814, 0.9317447198547334, 1.0883438689219551, 1.252148794071962, 0.9304411584181781]
[0.8920888562766902, 0.8459715322944616, 0.9383460822571011, 0.883301920698339, 0.8429355698888925]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9802701358569227 MAE: 0.871564396495808 Hidden channels: 128 threshold: 16
[0.8824971697310214, 0.9800072502080878, 1.0730807817620052, 0.9901755884258759, 0.9755898891576237]
[0.8433656307521948, 0.8804920716399907, 0.903200607196148, 0.8696367172332469, 0.8611269556574596]
Dataset: ../Datasets/fashion.pkl RMSE: 1.3732193797123406 MAE: 1.0404892353123656 Hidden channels: 128 threshold: 2
[1.2804424654816091, 1.308637059577696, 1.487745333519523, 1.3144283382305617, 1.4748437017523142]
[1.0080967057440904, 1.0187583787082841, 1.0796721256078423, 1.0205576523792417, 1.0753613141223703]
Dataset: ../Datasets/fashion.pkl RMSE: 1.3697706639313263 MAE: 1.0414171607948308 Hidden channels: 128 threshold: 3
[1.4378300956602517, 1.3126410033779752, 1.459241952599226, 1.2770471846863292, 1.3620930833328508]
[1.0681434520078663, 1.0199975775074563, 1.0750880227489303, 1.0036891912620725, 1.0401675604478293]
Dataset: ../Datasets/fashion.pkl RMSE: 1.300674795003233 MAE: 1.01124675349324 Hidden channels: 128 threshold: 4
[1.352282287285242, 1.3234648777978915, 1.2263557571279926, 1.2496085274264557, 1.3516625253785834]
[1.0259537574000868, 1.018629516601539, 0.986435662399492, 0.9912363908004804, 1.033978440264601]
Dataset: ../Datasets/fashion.pkl RMSE: 1.3103488547567932 MAE: 1.008746093410281 Hidden channels: 128 threshold: 5
[1.341640862079575, 1.3486837262750773, 1.2050440837896277, 1.4044580629600996, 1.2519175386795864]
[1.0108309817946426, 1.0328502562507065, 0.9744892498935342, 1.0401328739917366, 0.9854271051207848]
Dataset: ../Datasets/fashion.pkl RMSE: 1.247730709831934 MAE: 0.9868175189848488 Hidden channels: 128 threshold: 6
[1.1934850611662111, 1.252586700659847, 1.203211402805638, 1.2649081122077568, 1.324462272320217]
[0.9650229297963235, 0.9905034217400296, 0.9706896626178999, 0.9962442446454617, 1.0116273361245294]
Dataset: ../Datasets/fashion.pkl RMSE: 1.2160570455405142 MAE: 0.9708818139318737 Hidden channels: 128 threshold: 7
[1.2117977204192742, 1.1927127293809319, 1.192498937956807, 1.1890865212330972, 1.29418931871246]
[0.9712392904475946, 0.968280812132301, 0.9649209648300221, 0.9643453486860981, 0.9856226535633529]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1780523243721146 MAE: 0.9579511864182673 Hidden channels: 128 threshold: 8
[1.1883346002437538, 1.2031200796339487, 1.1835196619979171, 1.1339739117105594, 1.1813133682743937]
[0.9636998402374373, 0.959062411066077, 0.9674536130192919, 0.9412405382781511, 0.9582995294903786]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1681521443144454 MAE: 0.9549367425069615 Hidden channels: 128 threshold: 9
[1.1537572851220754, 1.2418521058138092, 1.1169764934890576, 1.14204784521709, 1.1861269919301938]
[0.946507932870127, 0.9843692734730839, 0.9342606440612293, 0.9489081269568711, 0.9606377351734964]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1322389222494635 MAE: 0.937132800592113 Hidden channels: 128 threshold: 10
[1.1490548003574224, 1.0752775068600404, 1.216850240788508, 1.1478361636314596, 1.0721758996098871]
[0.9422840660413445, 0.9148947697744118, 0.9679586920743256, 0.9453758036873166, 0.9151506713831666]
Dataset: ../Datasets/fashion.pkl RMSE: 1.0731087911627444 MAE: 0.9088528647223771 Hidden channels: 128 threshold: 11
[1.099985304761625, 1.051104485574424, 1.1161658000027768, 1.0164308361347987, 1.0818575293400985]
[0.9179385227119916, 0.9047959045112679, 0.9204034180680106, 0.8944384082517235, 0.9066880700688922]
Dataset: ../Datasets/fashion.pkl RMSE: 1.0982842497770642 MAE: 0.9225418715747559 Hidden channels: 128 threshold: 12
[1.0866946498717933, 1.067081836818795, 1.141231419530655, 1.089442318596368, 1.1069710240677102]
[0.9174669263504538, 0.9106164636803669, 0.9352337612215079, 0.9181400726770695, 0.9312521339443807]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1060706619881695 MAE: 0.9227396079884954 Hidden channels: 128 threshold: 13
[1.1677657394579686, 1.1408307479845836, 1.14136712032299, 0.9939529375043045, 1.0864367646710003]
[0.939225780833197, 0.928771883361217, 0.9422802465947455, 0.8860914360819373, 0.9173286930713807]
Dataset: ../Datasets/fashion.pkl RMSE: 1.112819817909021 MAE: 0.9255386661030391 Hidden channels: 128 threshold: 14
[1.0754194128637227, 1.2237803921420518, 1.0999125485864827, 1.1389338680199552, 1.0260528679328913]
[0.9035848343341938, 0.9660242053228474, 0.9204991276393093, 0.9415433514204763, 0.8960418117983681]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1090286622485521 MAE: 0.9231264244507276 Hidden channels: 128 threshold: 15
[1.1286057113727108, 1.0821984756676755, 1.065401297168569, 1.148580637985944, 1.120357189047862]
[0.9308456180564602, 0.9197032391570745, 0.8935087109610028, 0.9329918327005691, 0.9385827213785315]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1001468288580103 MAE: 0.9214059623772609 Hidden channels: 128 threshold: 16
[1.0438782971082796, 1.1520314018157864, 1.0691599813093982, 1.1666970333120592, 1.0689674307445276]
[0.905445418977849, 0.9345183538059693, 0.9088125842055438, 0.946607406336383, 0.9116460485605596]
    '''