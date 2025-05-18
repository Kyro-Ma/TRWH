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
    num_users_train = data_train['user'].num_nodes
    num_users_test = data_test['user'].num_nodes
    item_features_dim = data_train['item'].x.size(1)

    # Instantiate the model
    model = HeteroGNN(num_users_train, hidden_channels, item_features_dim).to(device)

    # Training process
    learning_rate = 0.001
    num_epochs = 900
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        out_dict = model(
            {
                'user': torch.arange(num_users_train).to(device),
                'item': data_train['item'].x.to(device)
            },
            data_train.edge_index_dict
        )
        user_out = out_dict['user'].to(device)
        user_indices = data_train['user', 'rates', 'item'].edge_index[0]
        predicted_ratings = model.lin(user_out[user_indices]).squeeze()
        loss = criterion(predicted_ratings, data_train['user', 'rates', 'item'].edge_attr.squeeze())
        loss.backward()
        optimizer.step()
        if loss.item() < 0.05:
            break
        # print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

    # Evaluation
    model.eval()
    with torch.no_grad():
        out_dict = model(
            {
                'user': torch.arange(num_users_test).to(device),
                'item': data_test['item'].x.to(device)
            },
            data_test.edge_index_dict
        )
        user_out = out_dict['user']
        user_indices = data_test['user', 'rates', 'item'].edge_index[0]
        predicted_ratings = model.lin(user_out[user_indices]).squeeze().tolist()

    # print(calculate_RMSE(predicted_ratings, testing_data['rating'].tolist()))
    # print(calculate_MAE(predicted_ratings, testing_data['rating'].tolist()))

    return predicted_ratings


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
    threshold_for_beauty = [4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
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

    for df_path, meta_df_path, w2vec_path, threshold in zip(df_path_list, meta_df_path_list, w2vec_path_list,
                                                            threshold_list):
        # if count == 0:
        #     count += 1
        #     continue

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
            w2v_model = Word2Vec(tokenized_titles, vector_size=vector_size, window=window, min_count=min_count,
                                 workers=workers)
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

                # Concatenate all the selected chunks for training
                result = train_and_evaluate(
                    pd.concat(train_chunks),
                    chunks[(i + num_folds - 1) % num_folds],  # Validation chunk
                    items_dict
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

        temp = [
            ['RMSE'] + RMSE_list,
            ['MAE'] + MAE_list
        ]

        # Create DataFrame
        df = pd.DataFrame(temp)

        if 'beauty' in df_path:
            output_path = f'../Datasets/Word2vec+HeteroGNN+InteractionFilter_hidden_channels={hidden_channels}_beauty.xlsx'
        else:
            output_path = f'../Datasets/Word2vec+HeteroGNN+InteractionFilter_hidden_channels={hidden_channels}_fashion.xlsx'
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        df.to_excel(
            output_path,
            index=False, header=False
        )
        print()


'''
Dataset: ../Datasets/beauty.pkl RMSE: 1.172464125991763 MAE: 0.9523033694823034 Hidden channels: 16 threshold: 4
[1.2052780826765523, 1.2176845591191379, 1.1333421792288898, 1.15504778522111, 1.1509680237131252]
[0.9636615921649605, 0.9719256780605174, 0.9415802882118541, 0.9426089239126346, 0.9417403650615507]
Dataset: ../Datasets/beauty.pkl RMSE: 1.103702530374727 MAE: 0.9238976731639845 Hidden channels: 16 threshold: 5
[1.088397948560498, 1.159957225745971, 1.0747134654791675, 1.088942658265158, 1.106501353822841]
[0.9175149279568695, 0.9426815076663402, 0.9075519470530881, 0.9204473902542162, 0.9312925928894087]
Dataset: ../Datasets/beauty.pkl RMSE: 1.30654031641686 MAE: 0.992785512201082 Hidden channels: 16 threshold: 6
[1.0830523090305928, 2.2867633392147626, 1.1221809077441234, 1.0310730009760696, 1.0096320251187512]
[0.9170171417697959, 1.3113613393107275, 0.9453531929671484, 0.8904311561540471, 0.8997647308036917]
Dataset: ../Datasets/beauty.pkl RMSE: 1.042342944734226 MAE: 0.9050649994235471 Hidden channels: 16 threshold: 7
[0.9906226310701765, 1.2448474421670939, 1.0267963479010291, 0.9368528469972826, 1.012595455535548]
[0.881514807668351, 0.9988838042804008, 0.890594819423868, 0.8657605904174643, 0.8885709753276508]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9769650438186069 MAE: 0.877808216041242 Hidden channels: 16 threshold: 8
[1.0189959410939657, 0.984374795196589, 0.9448309774370001, 0.9258394014919782, 1.0107841038735021]
[0.9035044182353856, 0.8720852810817817, 0.8672426513158623, 0.853409565205014, 0.8927991643681671]
Dataset: ../Datasets/beauty.pkl RMSE: 1.0408387302797848 MAE: 0.9095292876698025 Hidden channels: 16 threshold: 9
[1.021377792767993, 0.9897918292255071, 1.0806096211874665, 1.066411827022744, 1.0460025811952132]
[0.9016867763420334, 0.8872360814054263, 0.9262805780400873, 0.9254638105063824, 0.9069791920550831]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9645031485685778 MAE: 0.8717917204717087 Hidden channels: 16 threshold: 10
[0.924010859545905, 0.964682742911041, 0.932762428439801, 0.94822940053236, 1.0528303114137816]
[0.8667712590902957, 0.8710918228788792, 0.8624082008991817, 0.8577977088543706, 0.9008896106358167]
Dataset: ../Datasets/beauty.pkl RMSE: 1.546301134414973 MAE: 1.0707515198209994 Hidden channels: 16 threshold: 11
[0.9666082857537969, 2.2366137102592867, 0.9490448162026193, 2.625964850042394, 0.9532740098167684]
[0.8681773012947178, 1.3649616957034518, 0.8668132605496155, 1.3926314438911267, 0.8611738976660848]
Dataset: ../Datasets/beauty.pkl RMSE: 1.0020563758082512 MAE: 0.8893119830018573 Hidden channels: 16 threshold: 12
[1.1195084443989787, 0.9669378064820169, 1.0702269569256584, 0.9167086994560791, 0.936899971778522]
[0.9575627135935755, 0.8672682741648966, 0.9088634595167183, 0.8488863704143322, 0.8639790973197641]
Dataset: ../Datasets/beauty.pkl RMSE: 1.3414593880175163 MAE: 1.0261329954839522 Hidden channels: 16 threshold: 13
[1.4776984856905422, 1.014076374652701, 1.3936746989352342, 0.9665477011413428, 1.8552996796677617]
[1.0966094169903364, 0.8847494510312741, 1.0715601195863234, 0.8519471243021182, 1.225798865509708]
Dataset: ../Datasets/beauty.pkl RMSE: 1.5912713744005038 MAE: 1.0810493180672616 Hidden channels: 16 threshold: 14
[2.6342770441179617, 1.0734185529292963, 0.9168026043145642, 2.3326913906968074, 0.9991672799438893]
[1.3783637901336039, 0.9080492995223938, 0.8487062805901571, 1.3735314001039403, 0.8965958199862131]
Dataset: ../Datasets/beauty.pkl RMSE: 1.753900479806172 MAE: 1.0478046439835453 Hidden channels: 16 threshold: 15
[1.6143159918788712, 1.0469257662753626, 2.2432691157852576, 2.905204413399096, 0.9597871116922715]
[1.109747131231084, 0.9166743268131454, 1.3575239551810583, 0.9876502396185576, 0.8674275670738819]
Dataset: ../Datasets/beauty.pkl RMSE: 1.9281651647658105 MAE: 1.181729063300309 Hidden channels: 16 threshold: 16
[1.5335544283259723, 0.9432386223260728, 1.239676958420859, 1.7615232489095467, 4.162832565846601]
[1.1144201762976536, 0.8767179360672323, 0.9832625974256758, 1.1794842912056132, 1.7547603155053701]

Dataset: ../Datasets/beauty.pkl RMSE: 1.1903379945391173 MAE: 0.9580392424024442 Hidden channels: 64 threshold: 4
[1.1674456621633278, 1.197995202511652, 1.1650679555139871, 1.157976434339932, 1.2632047181666872]
[0.9485798279150187, 0.9617855694345875, 0.9537622576190381, 0.9374119682221357, 0.9886565888214405]
Dataset: ../Datasets/beauty.pkl RMSE: 1.1153270620269975 MAE: 0.9294193929154524 Hidden channels: 64 threshold: 5
[1.0974726225333415, 1.1561166163556802, 1.0972923559612764, 1.0960086612209696, 1.1297450540637193]
[0.9217037365689772, 0.9415494911131013, 0.9153144019263092, 0.9249773967064029, 0.9435519382624712]
Dataset: ../Datasets/beauty.pkl RMSE: 1.0652367159657188 MAE: 0.9065442357580361 Hidden channels: 64 threshold: 6
[1.1123475636792233, 1.0582541659677367, 1.0824962103841391, 1.0586400038820292, 1.0144456359154654]
[0.921759370103264, 0.9005235118876598, 0.9069165070933101, 0.9088950398821267, 0.8946267498238202]
Dataset: ../Datasets/beauty.pkl RMSE: 0.993371104952427 MAE: 0.8795176117078796 Hidden channels: 64 threshold: 7
[0.985173592157248, 1.0181161054092043, 1.0244121569396938, 0.956797143683429, 0.9823565265725598]
[0.8822894646315015, 0.884288548277679, 0.8849782512935015, 0.8690284650556218, 0.8770033292810946]
Dataset: ../Datasets/beauty.pkl RMSE: 0.986504723192982 MAE: 0.8767993915551884 Hidden channels: 64 threshold: 8
[0.9872567601004378, 0.964444180888393, 1.0162056425645591, 0.9436928250930872, 1.020924207318433]
[0.8806183018257606, 0.8610401535315878, 0.8839845767264753, 0.864111515068802, 0.8942424106233156]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9659227670383821 MAE: 0.867128667697622 Hidden channels: 64 threshold: 9
[0.9755979206424404, 0.9903122224223021, 0.9681193354897982, 0.9611677104729329, 0.934416646164437]
[0.8802170666057204, 0.8752031891022385, 0.8710141036390784, 0.8540667372090148, 0.8551422419320582]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9654139909365795 MAE: 0.8678217963411813 Hidden channels: 64 threshold: 10
[0.99129814166214, 0.895451443873321, 0.996276593405301, 0.9489807079321496, 0.9950630678099869]
[0.8848653450820173, 0.840126153196084, 0.8778557142647159, 0.8577647072746594, 0.8784970618884304]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9203803169206186 MAE: 0.8549333597646426 Hidden channels: 64 threshold: 11
[0.9077214190874291, 0.899185500107496, 0.9206932637272873, 0.9495258409884383, 0.9247755606924425]
[0.8484190926830396, 0.8456860227640138, 0.8595605670081695, 0.8670660616394069, 0.8539350547285832]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9591024723013197 MAE: 0.8636900240282678 Hidden channels: 64 threshold: 12
[0.9496498401355266, 1.0749574535545106, 0.9482697020661149, 0.9344375934360695, 0.8881977723143771]
[0.8598490112119145, 0.9191073095359414, 0.8472203833607432, 0.8441078096865164, 0.8481656063462234]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9645671473642989 MAE: 0.8703240625087332 Hidden channels: 64 threshold: 13
[0.9658255349144332, 0.9478394509078011, 0.9282637884372597, 0.934283977431575, 1.046622985130426]
[0.8765653073991523, 0.8510223281836818, 0.8610176571614085, 0.8671266555600631, 0.8958883642393602]
Dataset: ../Datasets/beauty.pkl RMSE: 1.0304157870920374 MAE: 0.896186829484026 Hidden channels: 64 threshold: 14
[0.993947967377071, 1.039331640037355, 0.8874425472905783, 1.3559516664730382, 0.8754051142821453]
[0.8738944710403413, 0.881142773351878, 0.8412088811758666, 1.0483651358417498, 0.836322886010294]
Dataset: ../Datasets/beauty.pkl RMSE: 1.2441483419384958 MAE: 0.9270866147905339 Hidden channels: 64 threshold: 15
[1.1296187908891153, 1.2589682046988802, 1.0438544395174232, 1.8186260615802252, 0.9696742130068352]
[0.9507469350395465, 0.9962596693334992, 0.9100491417475878, 0.9178013718674952, 0.8605759559645407]
Dataset: ../Datasets/beauty.pkl RMSE: 1.1359761601701834 MAE: 0.9218377709139818 Hidden channels: 64 threshold: 16
[0.9494445482345975, 0.9666900374456482, 1.7002875627366252, 1.0699166107828628, 0.9935420416511839]
[0.8769703200658683, 0.8755325535964233, 1.0695741471854914, 0.9030124877290389, 0.8840993459930869]

Dataset: ../Datasets/beauty.pkl RMSE: 1.2131387946506689 MAE: 0.9636325277647242 Hidden channels: 128 threshold: 4
[1.2149825260942524, 1.247229830549354, 1.1940594260830109, 1.2194939690190911, 1.1899282215076352]
[0.9680839543689179, 0.975639931879645, 0.9621993598441588, 0.9601008869711873, 0.9521385057597117]
Dataset: ../Datasets/beauty.pkl RMSE: 1.1266834970758104 MAE: 0.9324237647866079 Hidden channels: 128 threshold: 5
[1.1305930569077476, 1.1323641721497875, 1.114515711973884, 1.1560861834978577, 1.0998583608497747]
[0.932769642749919, 0.9247045698906695, 0.9281361264034923, 0.9509779311747659, 0.9255305537141929]
Dataset: ../Datasets/beauty.pkl RMSE: 1.109303571177882 MAE: 0.9247677773649426 Hidden channels: 128 threshold: 6
[1.1613476838945862, 1.0993094073320466, 1.157229887726767, 1.0900640114959028, 1.0385668654401072]
[0.9472113078698972, 0.9171832300962071, 0.9477017514726828, 0.9132147990274946, 0.8985277983584318]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9962871216698824 MAE: 0.8825241848850371 Hidden channels: 128 threshold: 7
[1.0096385154156908, 0.9974981658905806, 1.0440580308869352, 0.940055540587361, 0.9901853555688447]
[0.8897982868484307, 0.8766562957588446, 0.9027890488053697, 0.8653828683258271, 0.8779944246867137]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9625090361960298 MAE: 0.8700510051046393 Hidden channels: 128 threshold: 8
[0.9691785144444358, 0.9774485182527392, 0.9577886669628116, 0.928395923116362, 0.9797335582038007]
[0.8765506098142926, 0.8672264733221721, 0.8755310760321436, 0.8546356236022955, 0.8763112427522932]
Dataset: ../Datasets/beauty.pkl RMSE: 1.00428402285912 MAE: 0.8863712991853097 Hidden channels: 128 threshold: 9
[0.9732617246349491, 1.0458874800737057, 1.0295594486323552, 0.9610087271045374, 1.0117027338500535]
[0.8738111114412581, 0.9064164502103065, 0.9031714831285789, 0.8646331474634642, 0.8838243036829407]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9383677168999804 MAE: 0.8564724204223728 Hidden channels: 128 threshold: 10
[0.975501444209078, 0.8991478607643713, 0.8800832639926368, 0.9601628408610865, 0.9769431746727295]
[0.8839756437335762, 0.8409107042702963, 0.844717744465372, 0.8559936037311179, 0.8567644059115012]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9564949383609556 MAE: 0.8655983844295481 Hidden channels: 128 threshold: 11
[0.9037836433813431, 0.9376586834122683, 0.9499229135052881, 0.9755254669565933, 1.0155839845492847]
[0.8468250735965079, 0.8506523918297302, 0.8630430028719822, 0.8693374755604544, 0.898133978289066]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9407418293172757 MAE: 0.8599972279396537 Hidden channels: 128 threshold: 12
[0.9428201153543118, 0.9539562820643004, 0.9741801664830192, 0.9386882146691146, 0.8940643680156324]
[0.8717220725717076, 0.8687655296238775, 0.8594292011086324, 0.8555368558847718, 0.8445324805092792]
Dataset: ../Datasets/beauty.pkl RMSE: 0.9727150249954759 MAE: 0.8679809058237746 Hidden channels: 128 threshold: 13
[0.9868825484927776, 0.9639632328293812, 1.0376448081323189, 0.9002812931194308, 0.974803242403471]
[0.8748407817017463, 0.8591525726449599, 0.8959986950283804, 0.8418494787419525, 0.8680630010018341]
Dataset: ../Datasets/beauty.pkl RMSE: 1.0596791897909594 MAE: 0.8816877138293009 Hidden channels: 128 threshold: 14
[0.9567735743054471, 1.0355978507669361, 0.9713465893501333, 0.8908202402099049, 1.4438576943223747]
[0.8645990364801394, 0.8890709818757924, 0.8669578186514864, 0.8438620697958537, 0.9439486623432329]
Dataset: ../Datasets/beauty.pkl RMSE: 1.0047130159050377 MAE: 0.8803435481654214 Hidden channels: 128 threshold: 15
[1.0341897963874693, 0.9443677008552653, 1.020970715221264, 1.0195481039243133, 1.0044887631368768]
[0.894165849715899, 0.8627189003149474, 0.9003435324226468, 0.8653958430438112, 0.8790936153298026]
Dataset: ../Datasets/beauty.pkl RMSE: 1.1217149825117496 MAE: 0.9246956968861368 Hidden channels: 128 threshold: 16
[0.9265952940373423, 0.9789538884585571, 1.1988190807874994, 0.9871663140453893, 1.5170403352299606]
[0.8650308727616844, 0.869411263066099, 0.9303472152895846, 0.8727223322636579, 1.0859668010496584]
'''

'''
Dataset: ../Datasets/fashion.pkl RMSE: 1.2513428463038458 MAE: 0.9961429743065964 Hidden channels: 16 threshold: 4
[1.2420737475456693, 1.24436317653637, 1.2678971116387432, 1.2193160781151549, 1.2830641176832922]
[0.9932014211596103, 0.9944990448746718, 1.004340242501902, 0.9780246009284534, 1.0106495620683442]
Dataset: ../Datasets/fashion.pkl RMSE: 1.2398131261197793 MAE: 0.9933017407577276 Hidden channels: 16 threshold: 5
[1.2265632618637754, 1.278491175517295, 1.2490835718204667, 1.210363074486034, 1.2345645469113258]
[0.9881726577825274, 1.0100953632526974, 0.9969489929658113, 0.9862551931609764, 0.9850364966266258]
Dataset: ../Datasets/fashion.pkl RMSE: 1.2279638749677726 MAE: 0.9869241560425823 Hidden channels: 16 threshold: 6
[1.2781908012543677, 1.2041935419593448, 1.1872448437187917, 1.2430421473676876, 1.227148040538672]
[1.003618909727504, 0.9766247237356968, 0.967008852825591, 0.9966991964896286, 0.9906690974344917]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1882138345672772 MAE: 0.9649374513943132 Hidden channels: 16 threshold: 7
[1.1477867319187471, 1.1452776696813372, 1.2462267945272318, 1.2153338836841399, 1.1864440930249303]
[0.9476497297571564, 0.9531616746454492, 0.9850781589564493, 0.9786621390334574, 0.9601355545790534]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1575264710553614 MAE: 0.9514157949938571 Hidden channels: 16 threshold: 8
[1.182901101939788, 1.1814535423409227, 1.1589820349433884, 1.106795729932623, 1.1574999461200859]
[0.9623210887201374, 0.9564980295018338, 0.9607925155662795, 0.9319975650674108, 0.9454697761136241]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1383089943105869 MAE: 0.9416231349525234 Hidden channels: 16 threshold: 9
[1.1472511355713981, 1.1111697469428718, 1.1608166682702148, 1.162152673841664, 1.1101547469267854]
[0.9453476814379229, 0.9315510169554674, 0.9459334225984495, 0.9616843016001546, 0.9235992521706222]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1295436919860538 MAE: 0.9383965452508065 Hidden channels: 16 threshold: 10
[1.111151422988567, 1.0930299623021407, 1.1081364072170836, 1.161384532784346, 1.1740161346381308]
[0.9339383354047383, 0.9301883144416501, 0.9265570501506298, 0.9471772486707628, 0.9541217775862514]
Dataset: ../Datasets/fashion.pkl RMSE: 1.083555871356917 MAE: 0.9155659984333028 Hidden channels: 16 threshold: 11
[1.1051222489380026, 1.088904508016003, 1.1142015453989074, 1.0223686324300956, 1.0871824220015769]
[0.9229257777660784, 0.9151201526840641, 0.9279438979422266, 0.8951759827808236, 0.9166641809933216]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1273416590824181 MAE: 0.9349131372310218 Hidden channels: 16 threshold: 12
[1.0827710239145858, 1.078941217842518, 1.129303763077734, 1.138379710933124, 1.2073125796441293]
[0.9213275506166085, 0.9037061357007087, 0.9292192739680796, 0.9419391970333053, 0.9783735288364069]
Dataset: ../Datasets/fashion.pkl RMSE: 1.294052828533131 MAE: 0.9958808695466959 Hidden channels: 16 threshold: 13
[1.2891964355044698, 1.184173959802906, 1.1333636548877242, 1.7662656656427331, 1.097264426827821]
[0.9809854992008515, 0.940822805871475, 0.9358699521165353, 1.2110891785134859, 0.9106369120311315]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1055407953343808 MAE: 0.9283050145829179 Hidden channels: 16 threshold: 14
[1.1218894006881173, 1.0665043865138388, 1.1189048034576754, 1.2008451992058773, 1.0195601868063957]
[0.9319980501041876, 0.9105407327300089, 0.9434232763448448, 0.9689479923497477, 0.8866150213858006]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1016677680743991 MAE: 0.9280601073042085 Hidden channels: 16 threshold: 15
[1.1408716184931558, 1.0618065155392322, 0.9829332293398355, 1.1098380321992094, 1.212889444800562]
[0.9446135502545119, 0.9192948373109615, 0.8729754066248226, 0.9097603216721197, 0.9936564206586268]
Dataset: ../Datasets/fashion.pkl RMSE: 1.2265607198020343 MAE: 0.9942561676974242 Hidden channels: 16 threshold: 16
[1.5267490285273355, 1.1151209732871656, 1.2892098126152884, 1.2021258274655924, 0.9995979571147887]
[1.1419723223713085, 0.9162244983657655, 1.0680870931681388, 0.959542123084593, 0.8854548014973146]

Dataset: ../Datasets/fashion.pkl RMSE: 1.295355297682454 MAE: 1.0132400510630317 Hidden channels: 64 threshold: 4
[1.4506557563037272, 1.2511552746944279, 1.2548808561006606, 1.2195325083548554, 1.300552092958599]
[1.0747225603191854, 0.9955046110516871, 0.9963958412249583, 0.9843890331611562, 1.0151882095581712]
Dataset: ../Datasets/fashion.pkl RMSE: 1.2643576670398278 MAE: 0.9985644450749354 Hidden channels: 64 threshold: 5
[1.2979137491781076, 1.360768400700999, 1.2492374897941603, 1.1977642297203968, 1.2161044658054754]
[1.0100122024899232, 1.0327600889872797, 0.9955235395013812, 0.9763006898682637, 0.9782257045278288]
Dataset: ../Datasets/fashion.pkl RMSE: 1.220850382669718 MAE: 0.9785246093709681 Hidden channels: 64 threshold: 6
[1.2111497985675475, 1.236277755114025, 1.236735815101281, 1.2144520317149692, 1.2056365128507667]
[0.9726802314105589, 0.9830337682619616, 0.9846985385268557, 0.9770649228029727, 0.9751455858524916]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1958338520316005 MAE: 0.9672801460590801 Hidden channels: 64 threshold: 7
[1.1798746637407351, 1.1853409830219175, 1.252510124840733, 1.1805630403541538, 1.1808804482004622]
[0.9567626272074626, 0.9672774483423654, 0.9898153423461175, 0.9605158062440224, 0.9620295061554327]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1779683799515968 MAE: 0.9577493758506636 Hidden channels: 64 threshold: 8
[1.189723983920438, 1.2479325144216151, 1.1841622886796968, 1.0992799283841592, 1.1687431843520744]
[0.969090525003813, 0.9891808958722379, 0.9487091460557548, 0.93201742945346, 0.9497488828680523]
Dataset: ../Datasets/fashion.pkl RMSE: 1.142072383668522 MAE: 0.9446106862347273 Hidden channels: 64 threshold: 9
[1.1819476200249703, 1.151679111709892, 1.1213362128593234, 1.1216792004726746, 1.1337197732757505]
[0.9550936688649257, 0.9479063280288346, 0.9426765485079871, 0.9402618903711836, 0.937114995400706]
Dataset: ../Datasets/fashion.pkl RMSE: 1.148920825814781 MAE: 0.9476041507162245 Hidden channels: 64 threshold: 10
[1.1970295367115378, 1.1240614567833287, 1.1763698201069217, 1.1643927781644683, 1.0827505373076485]
[0.9818852309611249, 0.9367074370206445, 0.956049509490676, 0.9466398399037059, 0.9167387362049714]
Dataset: ../Datasets/fashion.pkl RMSE: 1.0990117567837099 MAE: 0.9223989879708133 Hidden channels: 64 threshold: 11
[1.1265235956688027, 1.0610468580038261, 1.145250642923066, 1.0570628065602543, 1.1051748807625994]
[0.9305093446595398, 0.9075843570741232, 0.9478094232364612, 0.9034773160474862, 0.9226144988364565]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1086462125943037 MAE: 0.9307306917405919 Hidden channels: 64 threshold: 12
[1.0794524706178443, 1.1196897587338415, 1.1609728542316082, 1.144758959236803, 1.0383570201514223]
[0.9241564485090531, 0.9292053570323547, 0.9491551501263585, 0.9469826825892146, 0.9041538204459784]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1087419367270563 MAE: 0.9242437807285893 Hidden channels: 64 threshold: 13
[1.2192368336685764, 1.0907330557648414, 1.1016968164484104, 1.0429651036334104, 1.0890778741200435]
[0.9647441824538618, 0.9079706147634722, 0.9304057639142947, 0.9061602440212394, 0.9119380984900778]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1283608197663935 MAE: 0.9336118362367154 Hidden channels: 64 threshold: 14
[1.1161478528285234, 1.0948107953248813, 1.2150597643572387, 1.177504215611291, 1.0382814707100327]
[0.920767546478708, 0.9120761994769933, 0.9819604992322163, 0.9527532594443713, 0.900501676551288]
Dataset: ../Datasets/fashion.pkl RMSE: 1.0888476459855423 MAE: 0.9168402585427634 Hidden channels: 64 threshold: 15
[1.0974011590533361, 1.0778789372042894, 1.0182926631191467, 1.113801812384936, 1.1368636581660025]
[0.9206465823645847, 0.9250664276487557, 0.880442239076869, 0.9113113014030538, 0.9467347422205536]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1249829090803083 MAE: 0.9305944432452596 Hidden channels: 64 threshold: 16
[1.1167927033484126, 1.145084509893567, 1.1548442143654227, 1.113607072124097, 1.0945860456700418]
[0.9292249254372517, 0.9236170900939442, 0.951925253614774, 0.9254262738915762, 0.9227786731887528]

Dataset: ../Datasets/fashion.pkl RMSE: 1.3128671688018816 MAE: 1.018716945883134 Hidden channels: 128 threshold: 4
[1.3294996494980855, 1.257118525412709, 1.3880408604052015, 1.283035581199274, 1.3066412274941381]
[1.0199994151547591, 1.0008888947397743, 1.0492526741089812, 1.0070507279758285, 1.016393017436327]
Dataset: ../Datasets/fashion.pkl RMSE: 1.2352777095686758 MAE: 0.9862697040835979 Hidden channels: 128 threshold: 5
[1.242948330131754, 1.244959990319719, 1.2024803786006546, 1.2133996434799101, 1.2726002053113405]
[0.9878287752860597, 0.9921055355858353, 0.9719450078080013, 0.9792199431857804, 1.0002492585523135]
Dataset: ../Datasets/fashion.pkl RMSE: 1.2323565435886483 MAE: 0.9836721186961045 Hidden channels: 128 threshold: 6
[1.2489156998404078, 1.1712390745666217, 1.2000421942552384, 1.2279677099303818, 1.3136180393505918]
[0.9894595077287159, 0.9599080797500321, 0.9686400085033497, 0.9833851715954577, 1.016967825902967]
Dataset: ../Datasets/fashion.pkl RMSE: 1.2559918947886735 MAE: 0.9921526109738699 Hidden channels: 128 threshold: 7
[1.2303873622029597, 1.2012642898430363, 1.2018765042659334, 1.4000074687337263, 1.2464238488977115]
[0.9828462676258749, 0.9758980993761982, 0.9679149237235525, 1.0530276171467536, 0.9810761469969697]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1911457938593109 MAE: 0.9640764577772073 Hidden channels: 128 threshold: 8
[1.202542923451745, 1.2190935178275515, 1.2641162666120331, 1.1098326609686744, 1.1601436004365508]
[0.9644293872716587, 0.9707893990322342, 0.9976377028835977, 0.9336447220969233, 0.9538810776016223]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1989736600208314 MAE: 0.9536134080359234 Hidden channels: 128 threshold: 9
[1.1813569839518974, 1.1478810832520754, 1.102385688660069, 1.1712928298964898, 1.3919517143436253]
[0.9595527046336669, 0.9464697930270959, 0.9318745143797748, 0.9431178776442827, 0.9870521504947971]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1348561394391257 MAE: 0.9399606516766081 Hidden channels: 128 threshold: 10
[1.1607007365897815, 1.084177426009133, 1.132012475465181, 1.193890801381501, 1.1034992577500313]
[0.957382342256066, 0.9153767428079445, 0.9373089952218304, 0.9624517020099229, 0.9272834760872769]
Dataset: ../Datasets/fashion.pkl RMSE: 1.0838195002401343 MAE: 0.9180390871263799 Hidden channels: 128 threshold: 11
[1.1143107704914053, 1.0771183417350485, 1.1081446248226696, 1.0402870588079776, 1.0792367053435712]
[0.9255572776633093, 0.9137492994686026, 0.9318669400558832, 0.9081783026544323, 0.9108436157896725]
Dataset: ../Datasets/fashion.pkl RMSE: 1.2512455805739586 MAE: 0.9393545457184211 Hidden channels: 128 threshold: 12
[1.1116875968478301, 1.1030077923396198, 1.1449037593137243, 1.7881787547123384, 1.1084499996562807]
[0.9343762777825066, 0.9160310377127406, 0.9316397427025995, 0.9872600543364026, 0.9274656160578567]
Dataset: ../Datasets/fashion.pkl RMSE: 1.1267801134104496 MAE: 0.9318380885124318 Hidden channels: 128 threshold: 13
[1.2738031389595657, 1.1319468580578889, 1.1007828078575732, 1.079695915897951, 1.047671846279269]
[0.9864219823545286, 0.9314702945284663, 0.9256125705606454, 0.9228404708355793, 0.8928451242829397]
Dataset: ../Datasets/fashion.pkl RMSE: 1.0835239729366242 MAE: 0.9122298229218874 Hidden channels: 128 threshold: 14
[1.1099857233917503, 1.0498621525730512, 1.0953047728277623, 1.1259490652554311, 1.036518150635126]
[0.9120679025402593, 0.8954772456202134, 0.9185543788387954, 0.9343776238044121, 0.9006719638057568]
Dataset: ../Datasets/fashion.pkl RMSE: 1.0880612878577651 MAE: 0.9180895288206452 Hidden channels: 128 threshold: 15
[1.0503392680152914, 1.0648699260371137, 1.011111352956089, 1.137998791533221, 1.1759871007471108]
[0.8975571871362427, 0.912705223972304, 0.8842102158863573, 0.9349592339009307, 0.9610157832073913]
Dataset: ../Datasets/fashion.pkl RMSE: 1.111984293102505 MAE: 0.9222039849447281 Hidden channels: 128 threshold: 16
[1.0214211428377107, 1.1654917771835966, 1.1192550238818915, 1.1636261000697348, 1.0901274215395917]
[0.8797349546454474, 0.9227677376529072, 0.9376058755836298, 0.9581490383534736, 0.9127623184881819]
'''