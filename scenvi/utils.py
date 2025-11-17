import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import scanpy as sc
import sklearn.neighbors
from clu import metrics
from flax import linen as nn
from flax import struct
from flax.training import train_state
from jax import random

from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm   
import scipy.sparse
from typing import Union
from scipy.sparse import issparse

class FeedForward(nn.Module):
    """
    :meta private:
    """

    n_layers: int
    n_neurons: int
    n_output: int

    @nn.compact
    def __call__(self, x):
        """
        :meta private:
        """

        n_layers = self.n_layers
        n_neurons = self.n_neurons
        n_output = self.n_output

        x = nn.Dense(
            features=n_neurons,
            dtype=jnp.float32,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )(x)
        x = nn.leaky_relu(x)
        x = nn.LayerNorm(dtype=jnp.float32)(x)

        for _ in range(n_layers - 1):

            x = nn.Dense(
                features=n_neurons,
                dtype=jnp.float32,
                kernel_init=nn.initializers.glorot_uniform(),
                bias_init=nn.initializers.zeros_init(),
            )(x)
            x = nn.leaky_relu(x) + x
            x = nn.LayerNorm(dtype=jnp.float32)(x)

        output = nn.Dense(
            features=n_output,
            dtype=jnp.float32,
            kernel_init=nn.initializers.glorot_uniform(),
            bias_init=nn.initializers.zeros_init(),
        )(x)

        return output


class CVAE(nn.Module):
    """
    :meta private:
    """

    n_layers: int
    n_neurons: int
    n_latent: int
    n_output_exp: int
    n_output_cov: int

    def setup(self):
        """
        :meta private:
        """

        n_layers = self.n_layers
        n_neurons = self.n_neurons
        n_latent = self.n_latent
        n_output_exp = self.n_output_exp
        n_output_cov = self.n_output_cov

        self.encoder = FeedForward(
            n_layers=n_layers, n_neurons=n_neurons, n_output=n_latent * 2
        )

        self.decoder_exp = FeedForward(
            n_layers=n_layers, n_neurons=n_neurons, n_output=n_output_exp
        )

        self.decoder_cov = FeedForward(
            n_layers=n_layers, n_neurons=n_neurons, n_output=n_output_cov
        )

    def __call__(self, x, mode="spatial", key=random.key(0)):
        """
        :meta private:
        """

        conf_const = 0 if mode == "spatial" else 1
        conf_neurons = jax.nn.one_hot(
            conf_const * jnp.ones(x.shape[0], dtype=jnp.int8), 2, dtype=jnp.float32
        )

        x_conf = jnp.concatenate([x, conf_neurons], axis=-1)
        enc_mu, enc_logstd = jnp.split(self.encoder(x_conf), 2, axis=-1)

        key, subkey = random.split(key)
        z = enc_mu + random.normal(key=subkey, shape=enc_logstd.shape) * jnp.exp(
            enc_logstd
        )
        z_conf = jnp.concatenate([z, conf_neurons], axis=-1)

        dec_exp = self.decoder_exp(z_conf)

        if mode == "spatial":
            dec_cov = self.decoder_cov(z)
            return (enc_mu, enc_logstd, dec_exp, dec_cov)
        return (enc_mu, enc_logstd, dec_exp)


@struct.dataclass
class Metrics(metrics.Collection):
    """
    :meta private:
    """

    enc_loss: metrics.Average
    dec_loss: metrics.Average
    enc_corr: metrics.Average


class TrainState(train_state.TrainState):
    """
    :meta private:
    """

    metrics: Metrics

def batch_matrix_sqrt(Mats):
    """
    :meta private:
    """

    e, v = np.linalg.eigh(Mats)
    e = np.where(e < 0, 0, e)
    e = np.sqrt(e)

    m, n = e.shape
    diag_e = np.zeros((m, n, n), dtype=e.dtype)
    diag_e.reshape(-1, n**2)[..., :: n + 1] = e
    return np.matmul(np.matmul(v, diag_e), v.transpose([0, 2, 1]))


def batch_knn(data, batch, k):
    """
    :meta private:
    """

    kNNGraphIndex = np.zeros(shape=(data.shape[0], k))

    for val in np.unique(batch):
        val_ind = np.where(batch == val)[0]

        batch_knn = sklearn.neighbors.kneighbors_graph(
            data[val_ind], n_neighbors=k, mode="connectivity", n_jobs=-1
        ).tocoo()
        batch_knn_ind = np.reshape(
            np.asarray(batch_knn.col), [data[val_ind].shape[0], k]
        )
        kNNGraphIndex[val_ind] = val_ind[batch_knn_ind]

    return kNNGraphIndex.astype("int")


def calculate_covariance_matrices(spatial_data, neighbor_mode, radius, kNN, niche_key, exp_data, spatial_key="spatial", batch_key: Union[str, int] = -1, batch_size=None):
    """
    Calculate covariance matrices for each cell/spot in spatial data using provided expression data
    
    :param spatial_data: AnnData object with spatial coordinates
    :param neighbor_mode: (str) 'knn' or 'radius' to define niche
    :param radius: (float) radius to define niche
    :param kNN: (int) number of nearest neighbours to define niche
    :param niche_key: (str) obs key name of niche label to restrict neighbors to same niche (default None, i.e. no restriction)
    :param exp_data: Pre-processed expression data to use for covariance calculation
    :param spatial_key: Key for spatial coordinates in obsm
    :param batch_key: Key for batch information in obs, or -1 for no batch
    :param batch_size: Number of cells to process at once (for memory efficiency)
    
    :return: 3D array of covariance matrices
    """
    
    coords = spatial_data.obsm[spatial_key]
    n_cells = coords.shape[0]
    neighbor_lists = [[] for _ in range(n_cells)]

    # Build neighbor graph
    if neighbor_mode == "knn":
        if batch_key == -1:
            kNNGraph = sklearn.neighbors.kneighbors_graph(
                coords,
                n_neighbors=kNN,
                mode="connectivity",
                n_jobs=-1,
            ).tolil()
            for i in range(n_cells):
                neighbor_lists[i] = np.array(kNNGraph.rows[i], dtype=int)
        else:
            for batch_value in np.unique(spatial_data.obs[batch_key]):
                batch_idx = np.where(spatial_data.obs[batch_key] == batch_value)[0]
                sub_coords = coords[batch_idx]
                knn_sub = sklearn.neighbors.kneighbors_graph(
                    sub_coords, n_neighbors=kNN, mode="connectivity", n_jobs=-1
                ).tolil()
                for row_i, cell_i in enumerate(batch_idx):
                    neighbor_lists[cell_i] = batch_idx[knn_sub.rows[row_i]]
                    
    elif neighbor_mode == "radius":
        if batch_key == -1:
            R_graph = sklearn.neighbors.radius_neighbors_graph(
                coords,
                radius=radius,
                mode="connectivity",
                include_self=False,
                n_jobs=-1,
            ).tolil()
            for i in range(n_cells):
                neighbor_lists[i] = np.array(R_graph.rows[i], dtype=int)
        else:
            for batch_value in np.unique(spatial_data.obs[batch_key]):
                batch_idx = np.where(spatial_data.obs[batch_key] == batch_value)[0]
                sub_coords = coords[batch_idx]
                R_graph = sklearn.neighbors.radius_neighbors_graph(
                    sub_coords,
                    radius=radius,
                    mode="connectivity",
                    include_self=False,
                    n_jobs=-1,
                ).tolil()

                for row_i, cell_i in enumerate(batch_idx):
                    neighbor_lists[cell_i] = batch_idx[R_graph.rows[row_i]]
    else:
        raise ValueError("neighbor_mode must be either 'knn' or 'radius'.")

    # Apply niche filtering if niche_key is provided
    if niche_key is not None:
        niche = np.asarray(spatial_data.obs[niche_key])
        filtered = []
        for i in range(n_cells):
            neigh = np.array(neighbor_lists[i], dtype=int)
            # keep only neighbors with same niche
            same = neigh[niche[neigh] == niche[i]]
            # fallback: if none exist, use original neighbors
            if same.size == 0:
                same = neigh
            filtered.append(same)
        neighbor_lists = filtered
    
    # Ensure each cell has at least one neighbor
    for i in range(n_cells):
        if len(neighbor_lists[i]) == 0:
            # fallback to nearest neighbor
            d = np.linalg.norm(coords - coords[i], axis=1)
            nearest = np.argsort(d)[1]  # skip self
            neighbor_lists[i] = np.array([nearest], dtype=int)
    
    # Build kNN graph index with sampling if necessary
    kNNGraphIndex = np.zeros((n_cells, kNN), dtype=int)
    for i in range(n_cells):
        neigh = neighbor_lists[i]
        # if neighbors fewer than kNN: sample with replacement
        if neigh.size >= kNN:
            kNNGraphIndex[i] = neigh[:kNN]
        else:
            fill = np.random.choice(neigh, size=kNN, replace=True)
            kNNGraphIndex[i] = fill
            
    # Get the global mean for each feature
    global_mean = exp_data.mean(axis=0)
    
    # Initialize the output covariance matrices
    n_cells = exp_data.shape[0]
    n_features = exp_data.shape[1]
    CovMats = np.zeros((n_cells, n_features, n_features), dtype=np.float32)
    
    # Process in batches if requested
    if batch_size is None or batch_size >= n_cells:
        # Process all cells at once
        print("Calculating covariance matrices for all cells/spots")
        DistanceMatWeighted = (
            global_mean[None, None, :]
            - exp_data[kNNGraphIndex[np.arange(n_cells)]]
        )
        
        CovMats = np.matmul(
            DistanceMatWeighted.transpose([0, 2, 1]), DistanceMatWeighted
        ) / (kNN - 1)
    else:
        # Process in batches to save memory
        batch_indices = np.array_split(np.arange(n_cells), np.ceil(n_cells / batch_size))
        
        for batch_idx in tqdm(batch_indices, desc="Calculating covariance matrices"):
            # Get neighbor indices for this batch
            batch_neighbors = kNNGraphIndex[batch_idx]
            
            # Calculate the distance matrices for this batch
            batch_distances = (
                global_mean[None, None, :]
                - exp_data[batch_neighbors]
            )
            
            # Calculate the covariance matrices for this batch
            batch_covs = np.matmul(
                batch_distances.transpose([0, 2, 1]), batch_distances
            ) / (kNN - 1)
            
            # Store the results
            CovMats[batch_idx] = batch_covs
    
    # Add a small regularization term to ensure positive definiteness
    reg_term = CovMats.mean() * 0.00001
    identity = np.eye(n_features)[None, :, :]
    CovMats = CovMats + reg_term * identity
    
    return CovMats


def niche_cell_type(spatial_data, neighbor_mode, kNN, radius, spatial_key="spatial", cell_type_key="cell_type",batch_key: Union[str, int] = -1,):
    """
    Compute local cell-type abundance in the spatial neighborhood (kNN or radius).
    
    Parameters
    ----------
    spatial_data : AnnData
    neighbor_mode : "knn" or "radius"
    kNN : number of neighbors for kNN mode
    radius : radius for radius mode
    spatial_key : obsm key for spatial coordinates
    cell_type_key : obs key for cell types
    batch_key : batch column or -1 for none
    
    Returns
    -------
    cell_type_niche : DataFrame, shape (n_cells, n_cell_types)
        Each row = abundance of each cell type in the neighborhood.
    """

    coords = spatial_data.obsm[spatial_key]
    n_cells = coords.shape[0]
    neighbor_lists = [[] for _ in range(n_cells)]

    if neighbor_mode == "knn":
        if batch_key == -1:
            kNNGraph = sklearn.neighbors.kneighbors_graph(
                coords,
                n_neighbors=kNN,
                mode="connectivity",
                n_jobs=-1
            ).tolil()
            for i in range(n_cells):
                neighbor_lists[i] = np.array(kNNGraph.rows[i], dtype=int)
        else:
            for batch_value in np.unique(spatial_data.obs[batch_key]):
                batch_idx = np.where(spatial_data.obs[batch_key] == batch_value)[0]
                sub_coords = coords[batch_idx]
                knn_sub = sklearn.neighbors.kneighbors_graph(
                    sub_coords, n_neighbors=kNN, mode="connectivity", n_jobs=-1
                ).tolil()
                for row_i, cell_i in enumerate(batch_idx):
                    neighbor_lists[cell_i] = batch_idx[knn_sub.rows[row_i]]

    elif neighbor_mode == "radius":
        if radius is None:
            raise ValueError("Radius must be provided when neighbor_mode='radius'")
        if batch_key == -1:
            R_graph = sklearn.neighbors.radius_neighbors_graph(
                coords,
                radius=radius,
                mode="connectivity",
                include_self=False,
                n_jobs=-1,
            ).tolil()
            for i in range(n_cells):
                neighbor_lists[i] = np.array(R_graph.rows[i], dtype=int)
        else:
            for batch_value in np.unique(spatial_data.obs[batch_key]):
                batch_idx = np.where(spatial_data.obs[batch_key] == batch_value)[0]
                sub_coords = coords[batch_idx]
                R_graph = sklearn.neighbors.radius_neighbors_graph(
                    sub_coords, radius=radius,
                    mode="connectivity",
                    include_self=False,
                    n_jobs=-1
                ).tolil()
                for row_i, cell_i in enumerate(batch_idx):
                    neighbor_lists[cell_i] = batch_idx[R_graph.rows[row_i]]
    else:
        raise ValueError("neighbor_mode must be 'knn' or 'radius'")
    
    for i in range(n_cells):
        if len(neighbor_lists[i]) == 0:
            d = np.linalg.norm(coords - coords[i], axis=1)
            nearest = np.argsort(d)[1] 
            neighbor_lists[i] = np.array([nearest], dtype=int)

    kNNGraphIndex = np.zeros((n_cells, kNN), dtype=int)
    for i in range(n_cells):
        neigh = neighbor_lists[i]
        if neigh.size >= kNN:
            kNNGraphIndex[i] = neigh[:kNN]
        else:
            kNNGraphIndex[i] = np.random.choice(neigh, size=kNN, replace=True)

    cell_types = np.asarray(spatial_data.obs[cell_type_key]).reshape(-1, 1)

    one_hot_enc = OneHotEncoder(sparse=True)
    cell_type_one_hot = one_hot_enc.fit_transform(cell_types)

    neigh_ohe = cell_type_one_hot[kNNGraphIndex]  # (N, kNN, n_types)

    neigh_sum = neigh_ohe.sum(axis=1)  # (N, n_types)

    if issparse(neigh_sum):
        neigh_sum = neigh_sum.A

    colnames = list(one_hot_enc.categories_[0])
    cell_type_niche = pd.DataFrame(neigh_sum, index=spatial_data.obs_names, columns=colnames)

    return cell_type_niche


def compute_covet(
    spatial_data, neighbor_mode="knn", k=8, radius=30.0, g=64, genes=None, spatial_key="spatial", batch_key="batch", 
    batch_size=None, use_obsm=None, use_layer=None, niche_key=None
):
    """
    Compute niche covariance matrices for spatial data, run with scenvi.compute_covet

    :param spatial_data: (anndata) spatial data, with an obsm indicating spatial location of spot/segmented cell
    :param neighbor_mode: (str) 'knn' or 'radius' to define niche (default 'knn')
    :param k: (int) number of nearest neighbours to define niche (default 8)
    :param radius: (float) radius to define niche when neighbor_mode is 'radius' (default 30.0)
    :param g: (int) number of HVG to compute COVET representation on (default 64)
    :param genes: (list of str) list of genes to keep for niche covariance (default [])
    :param spatial_key: (str) obsm key name with physical location of spots/cells (default 'spatial')
    :param batch_key: (str) obs key name of batch/sample of spatial data (default 'batch' if in spatial_data.obs, else -1)
    :param batch_size: (int) Number of cells/spots to process at once for large datasets (default None)
    :param use_obsm: (str) obsm key to use for COVET calculation instead of gene expression (e.g. 'X_pca', 'X_dc') (default None)
    :param use_layer: (str) layer to use for COVET calculation instead of log-transformed X (e.g. 'log', 'log1p') (default None)
    :param niche_key: (str) obs key name of niche label to restrict neighbors to same niche (default None, i.e. no restriction)
        
    :return COVET: niche covariance matrices
    :return COVET_SQRT: matrix square-root of niche covariance matrices for approximate OT
    :return CovGenes: list of genes selected for COVET representation (or feature names if using obsm)
    """

    genes = [] if genes is None else genes
    
    # Handle batch key
    if batch_key not in spatial_data.obs.columns:
        batch_key = -1
    
    # Determine data source: obsm, layer, or X
    if use_obsm is not None:
        if use_obsm not in spatial_data.obsm:
            raise ValueError(f"obsm key '{use_obsm}' not found in spatial_data.obsm")
        
        # Use the specified obsm embedding
        print(f"Computing COVET using obsm '{use_obsm}' with {spatial_data.obsm[use_obsm].shape[1]} dimensions")
        CovGenes = [f"{use_obsm}_{i}" for i in range(spatial_data.obsm[use_obsm].shape[1])]
        exp_data = spatial_data.obsm[use_obsm]
        
    else:
        # Select genes for covariance calculation
        if g == -1 or g >= spatial_data.shape[1]:
            # Use all genes
            CovGenes = spatial_data.var_names
            print(f"Computing COVET using all {len(CovGenes)} genes")
        else:
            # Check if highly variable genes need to be calculated
            if "highly_variable" not in spatial_data.var.columns:
                print(f"Identifying top {g} highly variable genes for COVET calculation")
                # Create a copy to avoid modifying the input
                spatial_data_copy = spatial_data.copy()
                
                # Determine appropriate layer for HVG calculation
                if use_layer is None:
                    if 'log' in spatial_data_copy.layers:
                        layer = "log"
                    elif 'log1p' in spatial_data_copy.layers:
                        layer = "log1p"
                    elif spatial_data_copy.X.min() < 0:
                        # Data is already log-transformed
                        layer = None
                    else:
                        X = spatial_data_copy.X
                        if issparse(X):
                            X_log = X.copy()
                            X_log.data = np.log1p(X_log.data)
                            spatial_data_copy.layers["log"] = X_log
                        else:
                            spatial_data_copy.layers["log"] = np.log1p(X)
                        layer = "log"
                else:
                    layer = use_layer
                
                # Calculate HVGs
                sc.pp.highly_variable_genes(
                    spatial_data_copy, 
                    n_top_genes=g, 
                    layer=layer if layer else None
                )
                # Get HVG names
                hvg_genes = spatial_data_copy.var_names[spatial_data_copy.var.highly_variable]
                if(len(hvg_genes) > g):
                    print(f"Found {len(hvg_genes)} HVGs")
            else:
                # HVGs already calculated
                hvg_genes = spatial_data.var_names[spatial_data.var.highly_variable]
                print(f"Using {len(hvg_genes)} pre-calculated highly variable genes for COVET")
            
            # Combine HVGs with manually specified genes
            CovGenes = np.asarray(hvg_genes)
            if len(genes) > 0:
                CovGenes = np.union1d(CovGenes, genes)
                print(f"Added {len(genes)} user-specified genes to COVET calculation")
                
            print(f"Computing COVET using {len(CovGenes)} genes")
        
        # Get expression data based on selected genes and specified layer
        if use_layer is not None:
            if use_layer not in spatial_data.layers:
                raise ValueError(f"Layer '{use_layer}' not found in spatial_data.layers")
            print(f"Using expression data from layer '{use_layer}'")
            exp_data = spatial_data[:, CovGenes].layers[use_layer].toarray() if scipy.sparse.issparse(spatial_data.layers[use_layer]) else spatial_data[:, CovGenes].layers[use_layer]
        else:
            # Default: log-transform X if needed
            if spatial_data.X.min() < 0:
                # Data is already log-transformed
                print("Using expression data from X (appears to be log-transformed)")
                exp_data = spatial_data[:, CovGenes].X.toarray() if scipy.sparse.issparse(spatial_data.X) else spatial_data[:, CovGenes].X
            else:
                print("Log-transforming expression data from X")
                exp_data = np.log(spatial_data[:, CovGenes].X.toarray() + 1) if scipy.sparse.issparse(spatial_data.X) else np.log(spatial_data[:, CovGenes].X + 1)
    
    # Calculate covariance matrices with batch processing
    COVET = calculate_covariance_matrices(
        spatial_data, neighbor_mode, radius, k, niche_key, exp_data, spatial_key=spatial_key, 
        batch_key=batch_key, batch_size=batch_size
    )
    
    # Calculate matrix square root
    if batch_size is None or batch_size >= COVET.shape[0]:
        print("Computing matrix square root...")
        COVET_SQRT = batch_matrix_sqrt(COVET)
    else:
        # Process matrix square root in batches too
        n_cells = COVET.shape[0]
        COVET_SQRT = np.zeros_like(COVET)
        
        # Split into batches
        batch_indices = np.array_split(np.arange(n_cells), np.ceil(n_cells / batch_size))
        
        for batch_idx in tqdm(batch_indices, desc="Computing matrix square roots"):
            # Process this batch of matrices
            batch_sqrt = batch_matrix_sqrt(COVET[batch_idx])
            COVET_SQRT[batch_idx] = batch_sqrt
    
    # Return results with proper types
    return (
        COVET.astype("float32"),
        COVET_SQRT.astype("float32"),
        np.asarray(CovGenes, dtype=str),
    )
