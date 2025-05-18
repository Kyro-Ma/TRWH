import sys
# from Codes.utils import load_dataset
# from utils import load_dataset
import json
import pickle  # Import pickle module for saving in pickle format
import multiprocessing
import pandas as pd


def load_dataset(file_path):
    df = pd.read_pickle(file_path)
    return df


# Function to create JSON for each user
def create_user_json_for_chunk(user_ids, df, meta_df):
    user_dict = {}

    # Iterate over each user in the chunk
    for user_id in user_ids:
        user_data = df[df['user_id'] == user_id]

        items = []
        # For each item interacted by the user
        for _, row in user_data.iterrows():
            item = {
                "title": row['title'],
                "review": row['text']
            }

            parent_asin = row['parent_asin']
            rows = meta_df.loc[meta_df['parent_asin'] == parent_asin, 'description']
            description = rows.iloc[0] if not rows.empty else None

            if len(description) > 0:
                item['description'] = description[0]
                # item['description'] = description
            else:
                item['description'] = None

            items.append(item)

        user_dict[user_id] = {
            "prompt": f"USER_ID: {user_id},\nPURCHASED ITEMS: \n[\n" + "\n".join([json.dumps(item) for item in items]) + "\n]"
        }

    return user_dict


# Function to split user list into chunks for multiprocessing
def chunkify(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


if __name__ == '__main__':
    # df_path = '../Datasets/beauty.pkl'
    # meta_path = '../Datasets/meta_beauty.pkl'
    # user_prompt_path = '../Datasets/beauty_user_prompt.pkl'
    df_path = '../Datasets/fashion.pkl'
    meta_path = '../Datasets/meta_fashion.pkl'
    user_prompt_path = '../Datasets/fashion_user_prompt.pkl'

    # Load datasets
    df = load_dataset(df_path)
    meta_df = load_dataset(meta_path)
    new_df = load_dataset(user_prompt_path)

    # Filter out users with fewer than 2 interactions; after filter: 298784, original: 2035490
    filtered_df = df.groupby('user_id').filter(lambda x: len(x) >= 2)

    # Get unique user IDs after filtering
    user_ids = filtered_df['user_id'].unique()

    # Split user IDs into chunks for parallel processing
    num_processes = multiprocessing.cpu_count()  # Number of processes to use
    user_chunks = chunkify(user_ids, len(user_ids) // num_processes)

    print(multiprocessing.cpu_count())
    # Create a pool of worker processes
    with multiprocessing.Pool(processes=num_processes) as pool:
        results = pool.starmap(create_user_json_for_chunk, [(chunk, df, meta_df) for chunk in user_chunks])

    # Combine results from all processes
    user_dict = {}
    for result in results:
        user_dict.update(result)

    # Save the user JSON to a file using pickle
    with open(user_prompt_path, 'wb') as f:  # Open in write-binary mode
        pickle.dump(user_dict, f)

