# TRWH
A Text-Driven Random Walk Heterogeneous GNN for Semantic-Aware Sparse Recommendation

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
Our experiments were conducted on both Linux and Windows platforms using Python 3.12. The LLMs-based experiments were conducted on a Linux system equipped with 8 40GB A100 GPUs, while the Word2Vec-based experiments were performed on a Windows system with a NVIDIA RTX 4080 GPU. The CUDA versions used were 12.0 on Linux and 12.6 on Windows. For PyTorch, we used version 2.6.0+cu118 on Linux and 2.7.0+cu126 on Windows. Regarding key hyperparameters, we experiment with learning rates of \{\(1 \times 10^{-3}\), \(1 \times 10^{-4}\), and \(1 \times 10^{-5}\)\} and train for \{\(300\), \(500\), \(800\), \(900\), \(1000\), \(1500\), \(2000\)\} epochs. The number of GNN layers is varied across \{\(1\), \(2\), \(3\)\}. We apply multiple hidden channels which are \{\(16\), \(32\), \(64\), \(128\)\}. We employ 5-fold cross-validation to ensure robust evaluation and apply an early stopping mechanism that halts training when the loss drops below \(0.05\). The Adam optimizer is used for model optimization, and Mean Squared Error (MSE) is adopted as the loss function. Our best results are obtained using a learning rate of \(1 \times 10^{-3}\), epochs \(\{900, 1000\}\), GNN layers \(1\), and hidden channels \(\{16, 32, 64, 128\}\). The choice of epoch count and hidden channel size depends on the specific model and dataset.
