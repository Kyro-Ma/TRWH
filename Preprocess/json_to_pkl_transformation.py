import json
import pandas as pd
import json
from tqdm import tqdm


def load_jsonl_data(file_path):
    """Load data from a JSONL (JSON Lines) file with a progress bar."""
    data = []

    # Open the file
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in tqdm(file, desc="Loading JSONL", unit="line"):
            data.append(json.loads(line))

    return pd.DataFrame(data)


if __name__ == '__main__':
    path = '../Datasets/Movies_and_TV.jsonl'
    meta_path = '../Datasets/meta_Movies_and_TV.jsonl'

    df = load_jsonl_data(path)
    meta_df = load_jsonl_data(meta_path)

    df.to_pickle('../Datasets/Movies_and_TV.pkl')
    meta_df.to_pickle('../Datasets/meta_Movies_and_TV.pkl')
