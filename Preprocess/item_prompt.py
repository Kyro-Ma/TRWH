import sys
import pandas as pd
# from Codes.utils import load_dataset
# from utils import load_dataset
import pickle
import multiprocessing
from multiprocessing import Pool
from functools import partial

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


def load_dataset(file_path):
    df = pd.read_pickle(file_path)
    return df


def clean_value(val, empty_check):
    try:
        null_val = pd.isnull(val)
        if isinstance(null_val, bool):
            if null_val:
                return None
        else:
            # For array-like values, check if all elements are null
            if null_val.all():
                return None
    except Exception:
        pass
    if empty_check(val):
        return None
    return val


def process_row(row, grouped_feedback):
    """
    Process a single row to generate the prompt.
    Returns a tuple (item_id, { "prompt": prompt }).
    """
    item_id = row['parent_asin']
    feedback_list = grouped_feedback.get(item_id, None)

    if feedback_list is not None:
        feedback_list = [f"\n{item}" for item in feedback_list]
        formatted_feedback = ''.join(feedback_list)
    else:
        formatted_feedback = None

    basic_info = {
        "main_category": row.get("main_category") if row.get("main_category") is not None else "None",
        "title": row.get("title") if row.get("title") is not None else "None",
        "description": {', '.join(row['description'])} if row.get("description") is not None else "None",
        "features": {', '.join(row['features'])} if row.get("features") is not None else "None",
        "details": row.get("details") if row.get("details") is not None else "None"
    }

    prompt = f"ITEM_ID: {item_id},\nBASIC INFORMATION: \n{basic_info}\nUSER FEEDBACK: {formatted_feedback}"

    return (item_id, {"prompt": prompt})


if __name__ == '__main__':
    # df_path = '../Datasets/beauty.pkl'
    # meta_path = '../Datasets/meta_beauty.pkl'
    # item_prompt_path = '../Datasets/beauty_item_prompt.pkl'

    df_path = '../Datasets/fashion.pkl'
    meta_path = '../Datasets/meta_fashion.pkl'
    item_prompt_path = '../Datasets/fashion_item_prompt.pkl'

    # Load datasets; parent_asin uses the asin from meta_df, number is: 826108
    df = load_dataset(df_path)
    meta_df = load_dataset(meta_path)
    item_prompt = load_dataset(item_prompt_path)

    # Filter out users with fewer than 2 interactions
    df = df.groupby('user_id').filter(lambda x: len(x) >= 2)

    # Apply cleaning rules on meta_df
    meta_df['price'] = meta_df['price'].apply(lambda x: None if pd.isnull(x) else x)
    meta_df['description'] = meta_df['description'].apply(
        lambda x: clean_value(x, lambda v: isinstance(v, list) and len(v) == 0))
    meta_df['categories'] = meta_df['categories'].apply(
        lambda x: clean_value(x, lambda v: isinstance(v, list) and len(v) == 0))
    meta_df['details'] = meta_df['details'].apply(
        lambda x: clean_value(x, lambda v: isinstance(v, dict) and len(v) == 0))
    meta_df['features'] = meta_df['features'].apply(
        lambda x: clean_value(x, lambda v: isinstance(v, list) and len(v) == 0))
    meta_df['store'] = meta_df['store'].apply(lambda x: None if pd.isnull(x) else x)

    # Group feedback by parent_asin from df
    grouped_feedback = df.groupby('parent_asin')['text'].apply(list).to_dict()

    # Convert meta_df rows to a list of dictionaries for processing
    meta_rows = meta_df.to_dict(orient='records')


    # Use multiprocessing to process rows concurrently
    with Pool(processes=multiprocessing.cpu_count()) as pool:
        results = pool.map(partial(process_row, grouped_feedback=grouped_feedback), meta_rows)

    # Combine the results into a dictionary where key is the item_id
    result = dict(results)
    print(len(result), result)

    # Save the result to a file using pickle (write in binary mode)
    with open(item_prompt_path, 'wb') as f:
        pickle.dump(result, f)
