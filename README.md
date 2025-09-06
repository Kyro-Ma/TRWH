# TRWH: A Text-Driven Random Walk Heterogeneous GNN for Semantic-Aware Sparse Recommendation

![Mainstructure of TRWH](mainstructure.png)

# Dataset
In this paper，we apply Amazon 2023 review dataset to evaluate our experiments - Amazon All_Beauty and Fashion.

Link: https://amazon-reviews-2023.github.io/

After downloading the datasets, including review data and meta data, create a directory - Datasets, put these four .jsonl files into it. Next, use the "json_to_pkl_transformation" file in Codes directory to convert .jsonl to .pkl.

# Environment setup
```bash
git clone https://github.com/Kyro-Ma/TRWH.git
cd TRWH

pip install -r requirements.txt

cd Preprocess
python json_to_pkl_transformation.py
```
Our experiments were conducted on both Linux and Windows platforms using Python 3.12. The LLMs-based experiments were conducted on a Linux system equipped with 8 40GB A100 GPUs, while the Word2Vec-based experiments were performed on a Windows system with a NVIDIA RTX 4080 GPU. The CUDA versions used were 12.0 on Linux and 12.6 on Windows. For PyTorch, we used version 2.6.0+cu118 on Linux and 2.7.0+cu126 on Windows.

# Training and Evaluation
In Codes directory, each file represents one of our proposed methods. 

For Word2Vec(W2V)-based approaches:
```bash
cd Codes

python <W2VHet.py/W2VRHet.py> # select one of methods
```

For LLM-based approaches, we firstly generate user and item prompts. Next, we generate their profiles with their prompts. Then, generate embeddings for users and items respectively before training. Finally, we input these embeddings into Heterogeneous graph for training:
```bash
cd Preprocess

python <user_prompt.py/item_prompt.py>
python <generate_user_profiles_DDP.py/generate_item_profiles_DDP.py>
python <generate_user_emb.py/generate_item_emb.py>

cd ../Codes

python <LLMHet.py/LLMRHet.py>
```


