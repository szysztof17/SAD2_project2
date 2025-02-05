v# using gdown
!gdown https://drive.google.com/uc?id=1yiFiwgm7Gy8_t_mhydZsnFyHDJAjC5tM -O GEX_train.h5ad # training dataset
!gdown https://drive.google.com/uc?id=1n2gLk95wRA55y84STdnJGlmjDqMjNfwV -O GEX_test.h5ad # testing dataset
!pip install anndata scanpy
import anndata as ad
import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import scipy
import numpy as np

adata_train = ad.read_h5ad("GEX_train.h5ad")
adata_test = ad.read_h5ad("GEX_test.h5ad")

print("Training dataset shape:", adata_train.shape)
print("Testing dataset shape:", adata_test.shape)

adata_train
adata_train.obs_names
adata_train.var_names

The train set contain 72208 observations and 5000 variables, the test set contains 18052 observation and 5000 variables. The observations are unique barcodes corresponding to a single cell each, and variables are gene names, likely cut down to 5000 top differentially expressed genes.
overlapping_cells = set(adata_train.obs_names) & set(adata_test.obs_names)
print(f"Number of overlapping cells: {len(overlapping_cells)}")

if overlapping_cells:
    print("Example overlapping cell IDs:", list(overlapping_cells)[:5])

train_only = set(adata_train.obs_names) - set(adata_test.obs_names)
test_only = set(adata_test.obs_names) - set(adata_train.obs_names)

print(f"Cells unique to train: {len(train_only)}")
print(f"Cells unique to test: {len(test_only)}")

set(adata_train.obs["is_train"])
n_train = sum(adata_train.obs["is_train"] == 'train')
n_test = sum(adata_train.obs["is_train"] == 'test')
n_holdput = sum(adata_train.obs["is_train"] == 'iid_holdout')
print(f"Number of train cells: {n_train}")
print(f"Number of test cells: {n_test}")
print(f"Number of holdout cells: {n_holdput}")
n_train = sum(adata_test.obs["is_train"] == 'train')
n_test = sum(adata_test.obs["is_train"] == 'test')
n_holdput = sum(adata_test.obs["is_train"] == 'iid_holdout')
print(f"Number of train cells: {n_train}")
print(f"Number of test cells: {n_test}")
print(f"Number of holdout cells: {n_holdput}")
adata_train.obs.head()
num_patients = adata_train.obs["DonorID"].nunique()
print(f"Number of patients: {num_patients}")

For each patient, we have information about their age, BMI, blood type, ethtnicity and race, gender and wheter their were a smoker or took any medication that could interfere as well as the site and batch that the sample comes from.
For each cell we have information about:

*   `n_genes_by_counts` or how many genes were expressed in the cell,
*   `pct_counts_mt`, how many genes were marked as mitochondrial,
*    `GEX_size_factors`, estimated size factor,
*    `GEX_phase`, estimated cell cycle phase.


print("Number of unique cell types:", adata_train.obs["cell_type"].nunique())

adata_train.obs["Site"].nunique()
set(adata_train.obs["Site"])
set(adata_train.obs["Samplename"])
table = "| Batch | Site | Donor  |\n|-----|------|--------|\n"

for entry in list(set(adata_train.obs["batch"])):
    _,site,_,  donor = list(entry)
    table += f"| {entry} | {site} | {donor} |\n"

print(table)

'aaa'.split()
len(set(adata_train.obs["Samplename"]))
set(adata_train.obs["cell_type"])
print("Number of batches:", adata_train.obs["batch"].nunique())

Batches are in the format dNsK, where dN means donor number N, and sK means site numer K. Batches are different groups of cells processed at different times or in different conditions, which may introduce technical variations (known as batch effect).



cell_counts = adata_train.obs["cell_type"].value_counts()
cell_counts.plot(kind="barh", figsize=(8, 6), title="Number of Observations per Cell Type")
plt.xlabel("Number of cells")
plt.ylabel("Cell Type")
plt.show()

cell_counts_per_donor = adata_train.obs.groupby(["DonorID", "cell_type"],observed=True).size().unstack(fill_value=0)

cell_counts_per_donor = cell_counts_per_donor[cell_counts_per_donor.sum().sort_values(ascending=False).index]


cell_counts_per_donor1 = cell_counts_per_donor
fig, ax = plt.subplots(figsize=(15, 20))
cell_counts_per_donor.plot(kind='barh', stacked=True, ax=ax, colormap="tab20")

plt.xlabel("Number of Cells")
plt.ylabel("Donor ID")
plt.title("Number of Observations for Each Cell Type and Each Patient")
plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()
plt.show()

cell_counts_per_donor = adata_train.obs.groupby(["DonorID", "cell_type"],observed=True).size().unstack(fill_value=0)

sorted_cell_types = cell_counts_per_donor.sum().sort_values(ascending=False).index
cell_counts_per_donor = cell_counts_per_donor[sorted_cell_types]

# Normalize to percentages
cell_counts_per_donor = cell_counts_per_donor.div(cell_counts_per_donor.sum(axis=1), axis=0) * 100


fig, ax = plt.subplots(figsize=(15, 20))
cell_counts_per_donor.plot(kind="barh", stacked=True, ax=ax, colormap="tab20")

plt.xlabel("Percentage of Cells (%)")
plt.ylabel("Donor ID")
plt.title("Proportional Distribution of Cell Types per Donor (Normalized)")
plt.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc="upper left")
plt.tight_layout()

plt.show()

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(20, 18))

# Plot 1: Bar plot of cell counts
cell_counts_per_donor1.plot(kind='barh', stacked=True, ax=axes[0], colormap="tab20")
axes[0].set_xlabel("Number of Cells", fontsize=16)
axes[0].set_ylabel("Donor ID", fontsize=16)
axes[0].set_title("Number of Observations for Each Cell Type and Each Patient", fontsize=22)

# Plot 2: Bar plot of normalized cell counts (percentages)
cell_counts_per_donor.plot(kind="barh", stacked=True, ax=axes[1], colormap="tab20")
axes[1].set_xlabel("Percentage of Cells (%)", fontsize=16)
axes[1].set_ylabel("Donor ID", fontsize=16)
axes[1].set_title("Proportional Distribution of Cell Types per Donor", fontsize=22)
for ax in axes:
    ax.get_legend().remove()

fig.legend(title="Cell Type", bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize='small')

plt.tight_layout()
plt.show()

total_elements = adata_train.layers["counts"].shape[0] * adata_train.layers["counts"].shape[1]

nonzero_count = adata_train.layers["counts"].nnz
zero_count = total_elements - nonzero_count

print("Total elements:", total_elements)
print("Nonzero elements:", nonzero_count)
print("Zero elements:", zero_count)
print(f"Fraction of nonzero elements: {nonzero_count / total_elements:.2%}")

expression_data = adata_train.layers['counts'].toarray()
#Compute mean and variance for each gene (across cells)

mean_counts = expression_data.mean(axis=0)
variance_counts = expression_data.var(axis=0)


df = pd.DataFrame({'mean_counts': mean_counts, 'variance_counts': variance_counts})

plt.figure(figsize=(8, 6))
plt.scatter(df['mean_counts'], df['variance_counts'])
plt.xscale('log')
plt.yscale('log')
plt.plot([1e-1, 1e2], [1e-1, 1e2], color='red', linestyle='--')  # Adding a red line with slope 1
plt.xlabel('Mean Counts')
plt.ylabel('Variance Counts')
plt.title('Mean vs Variance (Log-Scale)')
plt.grid(True, which="both", ls="--", linewidth=0.5)
plt.show()


import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns

# Extract raw count matrix from adata (assuming it's in sparse format)
raw_counts = adata_train.layers["counts"].toarray()

# Compute mean and variance for each gene (axis=1 means row-wise, i.e., gene-wise)
gene_means = np.mean(raw_counts, axis=1)
gene_variances = np.var(raw_counts, axis=1)

# Scatter plot of mean vs variance
plt.figure(figsize=(8,6))
plt.scatter(gene_means, gene_variances, alpha=0.5)
plt.plot([0, max(gene_means)], [0, max(gene_means)], 'r--', label="Poisson Expected (Var = Mean)")
plt.xlabel("Mean Expression")
plt.ylabel("Variance")
plt.title("Mean-Variance Relationship in scRNA-seq Data")
plt.legend()
plt.xscale("log")
plt.yscale("log")
plt.show()

# Check a few individual genes
np.random.seed(42)
selected_genes = np.random.choice(raw_counts.shape[0], size=3, replace=False)  # Pick 3 random genes

for gene_idx in selected_genes:
    observed_counts = raw_counts[gene_idx, :]
    
    # Fit a Poisson distribution based on the gene's mean
    lambda_hat = np.mean(observed_counts)
    poisson_dist = stats.poisson.pmf(np.arange(0, np.max(observed_counts) + 1), mu=lambda_hat)

    # Plot histogram of observed counts
    plt.figure(figsize=(6,4))
    sns.histplot(observed_counts, bins=30, kde=False, stat="probability", color="blue", label="Observed")
    
    # Overlay Poisson distribution
    plt.plot(np.arange(0, np.max(observed_counts) + 1), poisson_dist, 'ro-', label="Poisson Fit")
    
    plt.xlabel("Expression Counts")
    plt.ylabel("Probability")
    plt.title(f"Gene {gene_idx}: Observed vs Poisson")
    plt.legend()
    plt.show()

# Goodness-of-Fit Test: Chi-Square Test for a sample gene
gene_idx = selected_genes[0]  # Pick the first random gene
observed_counts = raw_counts[gene_idx, :]
lambda_hat = np.mean(observed_counts)
expected_poisson = stats.poisson.pmf(np.unique(observed_counts), mu=lambda_hat) * len(observed_counts)

chi2_stat, p_value = stats.chisquare(f_obs=np.bincount(observed_counts), f_exp=expected_poisson)

print(f"Chi-Square Test for Gene {gene_idx}: p-value = {p_value:.5f}")
if p_value < 0.05:
    print("Data significantly deviates from Poisson.")
else:
    print("No significant deviation from Poisson.")

max_val = adata_train.layers["counts"].max()
max_val
from scipy import ndimage
ndimage.histogram(adata_train.layers["counts"].data, 0, max_val, 25)


raw_counts = adata_train.layers["counts"].data
processed_counts = adata_train.X.data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(raw_counts, bins=100)
plt.title("Raw Counts Distribution")
plt.xlabel("Gene Expression Count")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(processed_counts, bins=100)
plt.title("Processed Counts Distribution")
plt.xlabel("Gene Expression Count")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(raw_counts,range=(5, 50), bins=20)
plt.title("Raw Counts Distribution, range from 5 to 50")
plt.xlabel("Gene Expression Count")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(processed_counts,range=(0, 10), bins=20)
plt.title("Processed Counts Distribution, range from 0 to 10")
plt.xlabel("Gene Expression Count")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


print("Raw: \nMin:", raw_counts.min(), "Max:", raw_counts.max(),
      "Mean:", raw_counts.mean(), "SD:", raw_counts.std())
print("Preprocessed: \nMin:", processed_counts.min(),
      "Max:", processed_counts.max(), "Mean:", processed_counts.mean(),
      "SD:", processed_counts.std())

raw_mtx = adata_train.layers["counts"]
processed_mtx = adata_train.X
scran = adata_train.layers["counts"] / adata_train.obs["GEX_size_factors"].values[:, None]

print("Preprocessed: \nMin:", scran.data.min(),
      "Max:", scran.data.max(), "Mean:", scran.data.mean(),
      "SD:", scran.data.std())

adata_train.obs['GEX_size_factors']

adata = adata_train.copy()
adata.X = adata_train.layers["counts"]

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)

print("Preprocessed: \nMin:", adata.X.data.min(),
      "Max:", adata.X.data.max(),
      "Mean:", adata.X.data.mean(),
      "SD:", adata.X.data.std())

adata = adata_train.copy()
adata.X = adata_train.layers["counts"]

sc.pp.normalize_total(adata, target_sum=1e4)
#sc.pp.log1p(adata)
print("Preprocessed: \nMin:", adata.X.data.min(),
      "Max:", adata.X.data.max(),
      "Mean:", adata.X.data.mean(),
      "SD:", adata.X.data.std())

adata = adata_train.copy()
adata.X = meme.copy().tocsr()
sc.pp.normalize_total(adata, target_sum=1e4)
#sc.pp.log1p(adata)
print("Preprocessed: \nMin:", adata.X.data.min(),
      "Max:", adata.X.data.max(),
      "Mean:", adata.X.data.mean(),
      "SD:", adata.X.data.std())
adata.X
adata = adata_train.copy()

size_factors = adata.obs['GEX_size_factors']

adata.X = adata.layers["counts"].multiply(1 /  size_factors.values.reshape(-1, 1))  
sc.pp.log1p(adata)

print("Preprocessed: \nMin:", adata.X.data.min(),
      "Max:", adata.X.data.max(),
      "Mean:", adata.X.data.mean(),
      "SD:", adata.X.data.std())
size_factors.values.reshape(-1, 1)
adata.obs['GEX_size_factors'].min()
!wget https://ftp.ncbi.nlm.nih.gov/geo/series/GSE194nnn/GSE194122/suppl/GSE194122%5Fopenproblems%5Fneurips2021%5Fcite%5FBMMC%5Fprocessed.h5ad.gz
When working with scRNA-Seq data, raw counts obtained experimentally need to be transformed for further analysis. Usually, the data is normalized using different methods (median count depth in basic Scanpy workflow, linear models using negative binomial distribution, bayesian and others) and log transformed.
!gunzip GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad.gz
adata_orig = ad.read_h5ad("GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")

adata_orig
raw_mtx = adata_orig.layers["counts"]
processed_mtx = adata_orig.X
print("Raw: \nMin:", raw_mtx.data.min(), "Max:", raw_mtx.data.max(),
      "Mean:", raw_mtx.data.mean(), "SD:", raw_mtx.data.std())
print("Preprocessed: \nMin:", processed_mtx.data.min(),
      "Max:", processed_mtx.data.max(), "Mean:", processed_mtx.data.mean(),
      "SD:", processed_mtx.data.std())

len(raw_mtx.data)
scran = adata_orig.layers["counts"] / adata_orig.obs["GEX_size_factors"].values[:, None]
adata_orig.layers["scran_normalization"] = scipy.sparse.csr_matrix(sc.pp.log1p(scran))

scran
print("Preprocessed: \nMin:", adata_orig.layers["scran_normalization"].data.min(),
      "Max:", adata_orig.layers["scran_normalization"].data.max(), "Mean:", adata_orig.layers["scran_normalization"].data.mean(),
      "SD:", adata_orig.layers["scran_normalization"].data.std())

scran = adata_orig.layers["counts"] / adata_orig.obs["GEX_size_factors"].values[:, None]

print("Preprocessed: \nMin:", scran.data.min(),
      "Max:", scran.data.max(), "Mean:", scran.data.mean(),
      "SD:", scran.data.std())

scran

### Task 2: **VAE implementation** (3 points)  

In this task you will implement Variational Autoencoder (Gaussian VAE) that can be trained on scRNA-seq data. The input and output should be the gene expression matrix (in this case a batch would consist of transcriptomic profiles for a subset of cells).

Remember that VAE needs to have a **stochastic Encoder** and **Decoder** and be trained with a **probabilistic loss**.

>**Notation**:
* $Z$ is a latent space
* $p(z)$ is *prior distribution* over the latent space
* $E(x)$ is an encoder
* $\phi$ are encoder network weights
* $D(z)$ is a decoder
* $\theta$ are decoder network weights

1. Implement `Encoder` and `Decoder` class in such a way that number of hidden layers, their sizes as well as size of latent space can be changed (you will need this to select the best size for the latent space). Each of them needs to have their `forward` method. Explain in your report what is output of Encoder and Decoder network in term of distributions (*Hint*: What are $q_\phi(z | x)$ and $p_\theta(x | z)$?)

2. Explain what *reparametrization trick* is. How and why is it used? Provide the mathematical formula and reasoning why it works.

3. Implement `VariationalAutoEncoder`class that combines `Encoder` and `Decoder` and includes:

  1. `reparameterize` method that implements *reparametrization trick* according to the formula from point 2.
  2. `sample_latent` method which accepts the original transcriptomic profile as input and outputs samples from the approximate posterior distribution $q_\phi(z|x)$.

---






### Task 3: **Kullback-Leibler divergence and ELBO** (3 points)

1. Explain what Kullback-Leibler divergence is and how it is in connected to VAE.

2. What is reconstruction loss? What should be the reconstruction loss in our case? Include a formula in the report.

3. What is ELBO? How is it connected to VAE? How is KL divergence connected to ELBO? How can ELBO be used to approximate $q_\phi(z | x)$ (math!)?

4. Implement `KL_divergence` function that computes the Kullback-Leibler divergence between $q_\phi(z | x)$ and $p(z)$. What is the assumed prior distribution of latent space $p(z)$ in Gaussian VAE? Include the formula your implementation is based on in your report.

5. Implement `ELBO` function. Include the formula your implementation is based on in your report.
  - *Hint 1*: See Appendix B from VAE paper: Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014, https://arxiv.org/abs/1312.6114,
  - *Hint 2*: Use the `KL_divergence` function from previous point.

6. ( *Optional* ) What is $\beta$-VAE? Explain how it differs from Vanilla VAE (math). Give an example (problem type, data type etc), where $\beta$-VAE would be better suited than Vanilla VAE. Give an example where it might not. Explain why.

---
### Task 4: **VAE training, latent space exploration and model selection** (4 points)

1. Prepare three pairs of training and testing datasets.

  1. raw count matrix (`adata.layers['counts']`)
  2. provided preprocessed matrix (`adata.X`)
  3. matrix processed as suggested in Task 1 point 5 (*Hint*: Copy matrix you will be processing further and save it in `adata.layers['counts_processed']`)

  For loading datsets for training and testing, [AnnLoader](https://anndata.readthedocs.io/en/latest/generated/anndata.experimental.AnnLoader.html) might be helpful.

2. Select a few different VAE architectures (that's where implementation from Task 2 point 1 comes in handy). Check at least 3 different sizes of latent space.

3. Train VAE models (different architectures and datasets). Note that testing dataset is in fact validation dataset and is used during training process.
  1. Verify the training procedure by showing learning curves. A learning curve for VAE usually plots the $-ELBO$ against epoch number.
  2. Break down $ELBO$ according to its decomposition and plot both losses separately.
    - *Hint 1.* What is reconstruction and regularization loss in VAE?
    - *Hint 2.* See Task 3 point 3 and 5.

4. Create a table showing different latent space sizes and different datasets and report the $-ELBO$, reconstruction and regularization loss on the test dataset for each model. Comment on the results.

5. Visualize latent space using t-SNE, UMAP or PCA.

  1. Explain which method you have chosen and how it works (no need for math, explain the intuition). You can visualize the sample from testing dataset, not the whole dataset, if the plots are hard to read.
  2. Color the plots by `adata.obs.cell_type`.
  3. Inspect those visualizations and comment on the results.

6. Select the final model. Explain the decision making process behind choosing the dataset and model's architecture (size of latent space). (*Hint*: Combine your observations from point 3 and 4 and come up with the consclusion.)

7. Visualize latent space of the selected model and color it by `adata.obs.DonorNumber`, `adata.obs.batch` and `adata.obs.Site`. Discuss what you see and include the figures in your report. (*Hint:* What are the batch effects?)

8. Compare visualization of latent space of your final model with selected dimentionality reduction method used on final test dataset. Color it by `adata.obs.cell_type`, `adata.obs.DonorNumber`, `adata.obs.batch` and `adata.obs.Site`. You can do it using [Scanpy](https://scanpy.readthedocs.io/en/stable/tutorials/basics/clustering.html): [t-SNE](https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pl.tsne.html), [UMAP](https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pl.umap.html), [PCA](https://scanpy.readthedocs.io/en/stable/api/generated/scanpy.pl.pca.html). Compare and comment on the results.

---
### Task 5: Write a report (3 points)

1. Write a final report. Below are suggested sections (you can structure it differently but it needs to be cohesive and easy to understand)
  1. Data Exploration.
  2. VAE theoretical background - here you can include answers to theoretical questions from Task 2 and 3.
  3. Methods
    - describe architecture and important implementation details
    - describe datasets
    - describe training parameters, loss functions and metrics used
  4. Results - here you can discuss model selection, results, visualizations, comparison.
  5. Conclusions


# using gdown
!gdown https://drive.google.com/uc?id=1yiFiwgm7Gy8_t_mhydZsnFyHDJAjC5tM -O GEX_train.h5ad # training dataset
!gdown https://drive.google.com/uc?id=1n2gLk95wRA55y84STdnJGlmjDqMjNfwV -O GEX_test.h5ad # testing dataset
import torch
import torch.nn as nn
class Encoder(nn.Module):
    def __init__(self,
                 # input, hidden and latent size
                 *args,
                 **kwargs
                 ):
        super().__init__()

        # initialize encoder structure
        pass

    def forward(self, x):
      # implement forward method
      pass
class Decoder(nn.Module):
    def __init__(self,
                 # latent size, hidden and output size
                 *args,
                 **kwargs
                 ):
        super().__init__()

        # initialize decoder structure
        pass

    def forward(self, z):
        # Implement forward method
        pass
class VAE(nn.Module):
    def __init__(self,
                 # encoder and decoder sizes
                 *args,
                 **kwargs
                 ):
        super().__init__()

        # initialize encoder and decoder
        pass

    def reparametrize(self,
                      # what else?
                      ):
        pass

    def sample_latent(self, x):
        pass

    def forward(self, x):
        # Implement forward method - pass input through
        pass


def KL_divergence():
  pass

def ELBO():
  pass
# Train model in training loop

# it might be helpful to implement train and test functions