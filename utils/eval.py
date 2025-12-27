import numpy as np


def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    p /= p.sum()
    q /= q.sum()
    return np.sum(p * np.log(p / q))


def js_divergence(p, q):
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    m = np.clip(m, epsilon, 1)
    kl_p_m = np.sum(p * np.log(p / m))
    kl_q_m = np.sum(q * np.log(q / m))
    js = 0.5 * (kl_p_m + kl_q_m)
    return js


def build_baseline_ave(x):
    if isinstance(x, list):
        x = np.concatenate(x, axis=0)
    x = x[:, 0, :, :]
    x_norm = x / np.max(x, axis=1)[:, np.newaxis, :]
    pred = np.mean(x_norm, axis=0)
    return pred


def build_baseline_zero(shape):
    return np.zeros(shape)


def calculate_adapted_pearson(
    array1, array2, adjacency_matrix, power=1, 
    weighting='equal_2_step', kernel='bisquare', bandwidth=2,
):
    """
    Calculate spatially-weighted correlation with value-dependent weighting.
    
    Computes a weighted bivariate correlation where spatial proximity and actual values 
    jointly determine the weight contribution. The weighting scheme w_ij(y_j) = w_ij^spatial * y_j^power
    allows higher values in array2 to have greater influence on the correlation.
    
    Parameters
    ----------
    array1 : np.array, shape (n, 1) or (n,)
        First variable (e.g., predictions)
    array2 : np.array, shape (n, 1) or (n,)
        Second variable (e.g., targets). Values from this array are used for weighting.
    adjacency_matrix : np.array, shape (n, n)
        Binary spatial adjacency matrix (1 for connected, 0 for not connected)
    power : float, default=1
    weighting : str, default='equal_2_step'
        Spatial weighting scheme:
        - 'equal_2_step': Binary adjacency-based weights (1 for neighbors, 0 otherwise).
          Includes self-connection (w_ii = 1).
        - 'geo_decay': Distance decay weights based on shortest network path.
          Uses kernel functions with specified bandwidth.
    kernel : str, default='bisquare'
        Kernel function for distance decay (only used when weighting='geo_decay'):
        - 'bisquare': Bi-square kernel (1 - (d/h)²)² for d < h
        - 'gaussian': Gaussian kernel exp(-d²/(2h²))
    bandwidth : float, default=2
        Bandwidth parameter for kernel functions (only used when weighting='geo_decay').
        Controls the rate of spatial decay.
        For example, if bandwidth=2, the location itself and the neighbors are used.
    
    Returns
    -------
    float
        Weighted correlation coefficient in range [-1, 1]
    
    Mathematical Formulation
    ------------------------
    The weighted correlation is computed as:
    
    $$I_{xy} = \\frac{\\sum_{i}\\sum_{j}w_{ij} \\cdot y_j^p \\cdot z_{x_i} 
    \\cdot z_{y_j}}{\\sqrt{\\sum_{i}\\sum_{j}w_{ij} \\cdot y_j^p \\cdot z_{x_i}^2} 
    \\cdot \\sqrt{\\sum_{i}\\sum_{j}w_{ij} \\cdot y_j^p \\cdot z_{y_j}^2}}$$
    
    where:
    - z_{x_i} = x_i - mean(x) (centered array1)
    - z_{y_j} = y_j - mean(y) (centered array2)
    - w_{ij} = spatial weights from adjacency or distance decay
    - y_j^p = value-based weighting component
    
    The normalization guarantees I_{xy} ∈ [-1, 1]
    
    Notes
    -----
    - Self-connections (i=j) are included with weight w_ii = 1 * y_i^power
    - Higher values in array2 contribute more to the correlation when power > 0
    - Designed for network-based spatial correlation analysis
    """

    # Flatten arrays if needed
    x = array1.flatten()
    y = array2.flatten()
    n = len(x)

    if weighting == 'equal_2_step':
        # combine spatial and value weights
        W = adjacency_matrix.copy().astype(float)
        W_weighted = np.zeros_like(W)
        for i in range(n):
            for j in range(n):
                if W[i, j] != 0:
                    W_weighted[i, j] = W[i, j] * (y[j] ** power)
                if i == j:  # consider self-connection
                    W_weighted[i, j] = 1 * (y[j] ** power)

    elif weighting == 'geo_decay':
        # Calculate shortest path distances (number of steps in network)
        # Replace 0s with inf for non-connected nodes, keep adjacency for connected
        from scipy.sparse.csgraph import shortest_path
        graph = adjacency_matrix.copy().astype(float)
        graph[graph == 0] = np.inf
        np.fill_diagonal(graph, 0)  # Distance to self is 0
        distance_matrix = shortest_path(csgraph=graph, directed=False, return_predecessors=False)
        if kernel == 'gaussian':
            # Gaussian kernel: w_ij = exp(-d_ij^2 / (2*h^2))
            W = np.exp(-distance_matrix ** 2 / (2 * bandwidth ** 2))
        elif kernel == 'bisquare':
            # Bi-square kernel
            W = np.zeros(distance_matrix.shape)
            mask = distance_matrix < bandwidth
            W[mask] = (1 - (distance_matrix[mask] / bandwidth) ** 2) ** 2
        else:
            raise ValueError(f"Unknown kernel: {kernel}. Use 'gaussian' or 'bisquare'")

        W_weighted = np.zeros_like(W)
        for i in range(n):
            for j in range(n):
                if W[i, j] != 0:
                    W_weighted[i, j] = W[i, j] * (y[j] ** power)
    
    else:
        raise ValueError(f"Unknown weighting method: {weighting}. Use 'equal_2_step' or 'geo_decay'")

    # Center the data (using original y values, not normalized)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    # Denominator based on normalization method
    weighted_var_x = 0
    weighted_var_y = 0
    for i in range(n):
        for j in range(n):
            weighted_var_x += W_weighted[i, j] * x_centered[i]**2
            weighted_var_y += W_weighted[i, j] * y_centered[j]**2
    denominator = np.sqrt(weighted_var_x * weighted_var_y)
    assert denominator != 0, "Denominator in weighted variance normalization cannot be zero"

    # Numerator: sum_i sum_j w_ij(y_j) * x_centered_i * y_centered_j
    numerator = 0
    for i in range(n):
        for j in range(n):
            numerator += W_weighted[i, j] * x_centered[i] * y_centered[j]

    I_weighted = numerator / denominator
    return I_weighted


def calculate_weighted_Pearson(pred, true, p=1):
    """
    Calculate weighted Pearson correlation coefficient.
    
    Parameters:
    -----------
    pred : array-like, shape (n,) or (n, 1)
        Predicted values
    true : array-like, shape (n,) or (n, 1)
        True values (used for weighting)
    p : float, default=1
        Power parameter for weighting (higher p gives more weight to high values)
    
    Returns:
    --------
    float
        Weighted Pearson correlation coefficient
    """
    # Flatten arrays to ensure 1D
    pred = np.array(pred).flatten()
    true = np.array(true).flatten()
    
    # Calculate weights: w_i = (true_i - min(true))^p
    weights = (true - np.min(true)) ** p
    
    # Handle edge case: if all weights are zero
    if np.sum(weights) == 0:
        raise ValueError("All weights are zero. Cannot compute weighted correlation.")
    
    # Calculate weighted means
    sum_weights = np.sum(weights)
    mean_pred_w = np.sum(weights * pred) / sum_weights
    mean_true_w = np.sum(weights * true) / sum_weights
    
    # Calculate weighted deviations
    dev_pred = pred - mean_pred_w
    dev_true = true - mean_true_w
    
    # Calculate weighted Pearson correlation
    numerator = np.sum(weights * dev_pred * dev_true)
    denominator = np.sqrt(np.sum(weights * dev_pred**2)) * np.sqrt(np.sum(weights * dev_true**2))
    
    # Handle edge case: if denominator is zero
    if denominator == 0:
        raise ValueError("Denominator is zero. Cannot compute correlation.")
    
    r_w = numerator / denominator
    
    return r_w


def cal_metrics(pred, true, label, verbose=0, adj_matrix=None):
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr, kendalltau, pearsonr

    rmse = np.sqrt(np.mean((pred - true) ** 2))
    mae = np.mean((np.abs(pred - true)))

    non_zero_mask = true != 0
    mape = np.mean(np.abs((true[non_zero_mask] - pred[non_zero_mask]) / (true[non_zero_mask] + 1e-10))) * 100

    # geo_pearson = calculate_geo_weighted_correlation(
    #     pred, true, adj_matrix, aggr='weighted_mean',
    #     bandwidth=2, kernel='bisquare', power=20
    # )

    adpated_pearson = calculate_adapted_pearson(pred, true, adj_matrix, power=20, weighting='equal_2_step')

    if verbose > 0:
        print(
            f'{label} MAE: {mae:.5f}; '
            f'{label} MAPE: {mape:.5f}; '
            f'{label} Adapted Pearson: {adpated_pearson:.5f}; ',
        )
    return {
        'MAE': mae, 'MAPE': mape, 'Adapted_Pearson': adpated_pearson,
    }


def aggr_metrics(metrics_list_dict, label):
    metric_names = list(metrics_list_dict[0].keys())
    metrics_list = [list(d.values()) for d in metrics_list_dict]
    metrics_array = np.array(metrics_list)
    metrics_mean = np.mean(metrics_array, axis=0)
    print(';\n'.join(f'{label} {n}: {v:.5f}' for n, v in zip(metric_names, metrics_mean)))
    return

