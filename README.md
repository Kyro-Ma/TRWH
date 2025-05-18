# TRWH
A Text-Driven Random Walk Heterogeneous GNN for Semantic-Aware Sparse Recommendation

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
```
Our experiments were conducted on both Linux and Windows platforms using Python 3.12. The LLMs-based experiments were conducted on a Linux system equipped with 8 40GB A100 GPUs, while the Word2Vec-based experiments were performed on a Windows system with a NVIDIA RTX 4080 GPU. The CUDA versions used were 12.0 on Linux and 12.6 on Windows. For PyTorch, we used version 2.6.0+cu118 on Linux and 2.7.0+cu126 on Windows.
