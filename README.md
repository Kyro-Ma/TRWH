# TRWH: A Text-Driven Random Walk Heterogeneous GNN for Semantic-Aware Sparse Recommendation

![Mainstructure of TRWH](mainstructure.png)

# Our results

<table>
  <tr>
    <td>

<!-- Fashion Table -->
<b>Table 1: Fashion Dataset</b>  
<i>[*] denotes our proposed methods</i>

<table>
<thead>
<tr><th>Model</th><th>RMSE</th><th>MAE</th></tr>
</thead>
<tbody>
<tr><td>P²MF</td><td>11.091</td><td>2.538</td></tr>
<tr><td>APAR</td><td>19.483</td><td>3.712</td></tr>
<tr><td>PTUPCDR</td><td>7.927</td><td>1.411</td></tr>
<tr><td>CDRIB</td><td>7.063</td><td>1.437</td></tr>
<tr><td>UniCDR</td><td>7.173</td><td>1.45</td></tr>
<tr><td>NMCDR</td><td>8.98</td><td>3.215</td></tr>
<tr><td>RealHNS</td><td>8.64</td><td>1.520</td></tr>
<tr><td>MAN</td><td>10.43</td><td>5.984</td></tr>
<tr><td>Homogeneous GNN</td><td>1.15</td><td>1.66</td></tr>
<tr><td>PEMF-CD</td><td>6.96</td><td>1.392</td></tr>
<tr><td><b>W2VRHet [*]</b></td><td>1.0731</td><td><b>0.9089</b></td></tr>
<tr><td><b>LLMRHet [*]</b></td><td>1.0698</td><td>0.9191</td></tr>
<tr><td><b>W2VHet [*]</b></td><td>1.0775</td><td>0.9105</td></tr>
<tr><td><b>LLMHet [*]</b></td><td><b>1.0604</b></td><td>0.9107</td></tr>
</tbody>
</table>

</td>
<td style="padding-left: 40px;">

<!-- Beauty Table -->
<b>Table 2: Beauty Dataset</b>  
<i>[*] denotes our proposed methods</i>

<table>
<thead>
<tr><th>Model</th><th>RMSE</th><th>MAE</th></tr>
</thead>
<tbody>
<tr><td>MF</td><td>1.1973</td><td>0.9461</td></tr>
<tr><td>MLP</td><td>1.3078</td><td>0.9597</td></tr>
<tr><td>P5</td><td>1.2843</td><td>0.8534</td></tr>
<tr><td>ChatGPT (few-shot)</td><td>1.0751</td><td><b>0.6977</b></td></tr>
<tr><td>Homogeneous GNN</td><td>1.18</td><td>1.69</td></tr>
<tr><td><b>W2VRHet [*]</b></td><td>0.9327</td><td>0.8496</td></tr>
<tr><td><b>LLMRHet [*]</b></td><td>0.9134</td><td>0.8533</td></tr>
<tr><td><b>W2VHet [*]</b></td><td>0.9204</td><td>0.8549</td></tr>
<tr><td><b>LLMHet [*]</b></td><td><b>0.8944</b></td><td>0.8421</td></tr>
</tbody>
</table>

</td>
  </tr>
</table>

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

# Training and Evaluation
In Codes directory, each file represents one of our proposed methods. 

```bash
cd Codes

python <LLMHet.py/LLMRHet.py/W2VHet.py/W2VRHet.py> # sekect one of methods
```
