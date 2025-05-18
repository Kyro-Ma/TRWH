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
        if counter % print_counter == 0:
            print(str(round(counter / rate_count_train, 2) * 100) + '%')

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
    print("Getting item features (training)")
    for value in iid_train.keys():
        if counter % print_counter == 0:
            print(str(round(counter / len(iid_train.keys()), 2) * 100) + '%')

        target = items_dict[value]
        temp = [target['average_rating'], target['rating_number']] + target['title'].tolist()
        item_features_train.append(temp)
        counter += 1

    counter = 0
    print("Getting item features (testing)")
    for value in iid_test.keys():
        if counter % print_counter == 0:
            print(str(round(counter / len(iid_test.keys()), 2) * 100) + '%')

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
        if counter % print_counter == 0:
            print(str(round(counter / len(iid_test.keys()), 2) * 100) + '%')

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

    print('train edge data finished')

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
        if counter % print_counter == 0:
            print(str(round(counter / rate_count_test, 2) * 100) + '%')

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

    print('test edge data finished')

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
    # print(item_features_dim)
    hidden_channels = 16

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
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item()}')

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

    print(calculate_RMSE(predicted_ratings, testing_data['rating'].tolist()))
    print(calculate_MAE(predicted_ratings, testing_data['rating'].tolist()))

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


def convert_rating(rating, threshold=3):
    if rating > threshold:
        return 1
    if rating < threshold:
        return -1
    return 0


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
    device = 'cuda'
    torch.cuda.manual_seed_all(42)  # If you're using GPU
    np.random.seed(42)
    df_path_list = [beauty_path, fashion_path]
    meta_df_path_list = [meta_beauty_path, meta_fashion_path]
    w2vec_path_list = [beauty_w2v_path, fashion_w2v_path]

    for df_path, meta_df_path, w2vec_path in zip(df_path_list, meta_df_path_list, w2vec_path_list):
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

        meta_df = meta_df.reset_index(drop=True)
        df = df.reset_index(drop=True)
        df = df[0: 500000]
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
        # w2v_model = Word2Vec(tokenized_titles, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
        # w2v_model.save(w2vec_path)

        # load trained model directly if model has trained
        w2v_model = Word2Vec.load(w2vec_path)

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
            if counter % print_counter == 0:
                print(str(round(counter / rate_count_train, 2) * 100) + '%')

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

        print(rmse_list)
        print(mae_list)

        with open('mae.pkl', 'wb') as f:
            pickle.dump(mae_list, f)
        with open('rmse.pkl', 'wb') as f:
            pickle.dump(rmse_list, f)

        gc.collect()
        torch.cuda.empty_cache()
    '''
    word2vec with random walk on beauty (RMSE:1.72, MAE: 1.17, 5-fold) 
    [1.6555746808521565, 1.7462304536947393, 1.7329464678495365, 1.713069681789971, 1.759225445570849]
    [1.1492184829078846, 1.187734466279874, 1.1766707302166426, 1.1694747221330386, 1.1855778047845364]
    
    (RMSE:1.68, MAE: 1.16, 5-fold) 
    [1.9042886728900281, 1.6444967762070444, 1.6756216837480216, 1.5937625187425517, 1.5792180945978807]
    [1.225920217738759, 1.1510981838047203, 1.162666599613265, 1.1322257271303442, 1.1218885683742752]
    
    (RMSE:1.49, MAE: 1.09, 5-fold)
    [1.3862719585872727, 1.5767801967856752, 1.6068082493986688, 1.4610927249259045, 1.4362896133584395]
    [1.054321642254241, 1.1189206665920437, 1.1248351730594153, 1.085249962594041, 1.072485031871283]

    (RMSE:1.66, MAE: 1.15, 10-fold) 
    [1.6057080334802, 1.5533700916596531, 1.7090691556446589, 1.6402362733741402, 1.711626990549515, 1.6982907223343042, 1.54692744714422, 1.7123085759261858, 1.819888732152192, 1.5600819519819806]
    [1.1383813404605363, 1.1235729602030364, 1.1718858040484528, 1.1412421680424036, 1.1679706645003227, 1.1666269619307108, 1.1131814976350947, 1.169629930638297, 1.1896046848065933, 1.1260507315989223]
    
    (RMSE:1.64, MAE: 1.14, 10-fold)
    [1.5353191373084212, 1.8171162615524676, 1.5620670833806842, 1.5899229154993508, 1.5690896627887934, 1.6735226969112946, 1.5884224677328798, 1.7205008578974461, 1.6743886841198228, 1.7024773516339322]
    [1.1139404694578021, 1.1980867825594346, 1.1226627046706137, 1.1284677455405407, 1.122583842646387, 1.1520308554763279, 1.131313922955473, 1.1601769006925726, 1.1552216732900273, 1.1643932670950996]
    
    (RMSE:1.33, MAE: 1.02, 10-fold data: 10000)
    [1.2956955543217, 1.2831660791044284, 1.3153794634840843, 1.571973826212158, 1.29051604139963, 1.3126006500151064, 1.306863530465597, 1.3103217104533753, 1.3594645897041395, 1.2914514203458434]
    [1.0097516940131714, 1.010722063861228, 1.0210147017430182, 1.0855255283880079, 1.0145843659512528, 1.0071242139949812, 1.0113994078490922, 1.0075043150446141, 1.029331298266985, 1.017147322182055]
    
    (RMSE:1.32, MAE: 1.02, 10-fold data: 500000)
    [1.3279592898926076, 1.367035893091074, 1.3609489200728389, 1.2803313107158387, 1.3505109056089426, 1.3064288773308466, 1.3193802707738442, 1.2877840769920006, 1.337463130988528, 1.240301770171301]
    [1.0150122662631655, 1.025784728486961, 1.0333229526991778, 0.9963882388478615, 1.0262433813395089, 1.0183260208786997, 1.034756555428653, 1.0076372002134664, 1.03599169946488, 0.9931480573286885]
    
    (RMSE:1.32 MAE: 1.09 5-fold)
    '''

    '''
    word2vec on fashion
    with one layer(word2vec with random walk): (RMSE:1.53, MAE: 1.10, 5-fold, embed title only)
    [1.5224945562186774, 1.535177416335963, 1.519304409884702, 1.5697144788413249, 1.5121540613613909]
    [1.0975376398862275, 1.0983460181324478, 1.1024195614892467, 1.1131063106053958, 1.103958303331436]
    
    (RMSE:1.56, MAE: 1.11, 5-fold, embed title and text)
    [1.5557399309949809, 1.5219900484613744, 1.6014298901813888, 1.54179165767563, 1.5705697262949603]
    [1.1137972536268477, 1.1066894025874048, 1.1301280239434566, 1.1026162867686837, 1.1136961245382417]
    
    (RMSE:1.62, MAE: 1.13, 10-fold)
    [1.6370284759982994, 1.521844546527771, 1.5175107017917513, 1.525935654892469, 2.11685562143817, 1.5778947823569986, 1.5517126832171086, 1.764430176939989, 1.4934930031555935, 1.5005086521397277]
    [1.143840462419231, 1.0995536395314502, 1.0993859511182946, 1.1022393117173341, 1.2671812369223738, 1.1271177788652034, 1.1160399474658345, 1.197763875778764, 1.099807993041382, 1.090860006415131]
    
    (RMSE:1.56, MAE: 1.11, 10-fold)
    [1.559614195026397, 1.497052655737119, 1.4839893943651232, 1.6368658255357094, 1.6979386102221352, 1.505303946536504, 1.6057244313954617, 1.524974267613384, 1.5741623827984617, 1.5325708514969454]
    [1.1138397531732545, 1.0907409694033532, 1.0862023457849868, 1.1245861149027145, 1.151808095659746, 1.10780891203306, 1.1223708036452937, 1.1052190548037288, 1.116542673583798, 1.1045792074699623]
    
    (RMSE:1.56, MAE: 1.11, 5-fold embed title and text both)
    [1.5557399309949809, 1.5219900484613744, 1.6014298901813888, 1.54179165767563, 1.5705697262949603]
    [1.1137972536268477, 1.1066894025874048, 1.1301280239434566, 1.1026162867686837, 1.1136961245382417]
    
    (RMSE:1.22, MAE: 0.97 10-fold data: 10000)
    [1.1893158789400446, 1.22273151410776, 1.3251573968842467, 1.1638039504219593, 1.2596348706448048, 1.1414011102814372, 1.2669284844794095, 1.1552059024064083, 1.193263916495218, 1.2443964905771054]
    [0.9526185977997463, 0.9874001115289477, 1.0153238941492868, 0.9490833247088428, 0.9932931934530951, 0.9436314143272224, 0.9887033321722157, 0.9617457540208634, 0.9553317053211743, 0.9837887856182218]
    
    (RMSE:1.56, MAE: 1.11 10-fold data: 500000)
    [1.7980821401511575, 1.4678977236581734, 1.4782659834636724, 1.5768742486202443, 1.4538554409040523, 1.4255336878010463, 1.66927760415751, 1.687974678520086, 1.6063022896306918, 1.4750891883138666]
    [1.1760678921754968, 1.0860383321431115, 1.0832976989124519, 1.121097015162774, 1.0772905190699158, 1.058295041066663, 1.1304848609339133, 1.1615351845378337, 1.1333560529493725, 1.0887528899120376]
    
    (RMSE:1.29, MAE: 1.00 5-fold data: 500000)
    [1.3157731883980426, 1.2349722640657048, 1.2387987977320933, 1.281970855890682, 1.354747774614867]
    [1.0043570833326172, 0.9814618126376276, 0.9814129913441235, 1.0019801405023752, 1.0219985753345906]
    '''
