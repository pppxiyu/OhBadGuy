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


def calculate_moran_bv_global(array1, array2, adjacency_matrix):
    import numpy as np
    from libpysal.weights import W
    from esda.moran import Moran_BV
    """
    Calculate bivariate spatial correlation (Moran's I) between two arrays.
    
    Parameters:
    -----------
    array1 : np.array
        First array, shape (n, 1) or (n,)
    array2 : np.array
        Second array, shape (n, 1) or (n,)
    adjacency_matrix : np.array
        Square adjacency matrix, shape (n, n)
    
    Returns:
    --------
    dict : Dictionary containing:
        - 'I': Bivariate Moran's I statistic
        - 'p_value': P-value from permutation test
        - 'z_score': Standardized z-score
    """
    # Flatten arrays if they're (n, 1) shaped
    x = array1.flatten()
    y = array2.flatten()
    
    # Convert adjacency matrix to PySAL weights object
    n = len(adjacency_matrix)
    neighbors = {}
    weights = {}
    
    for i in range(n):
        neighbors[i] = []
        weights[i] = []
        for j in range(n):
            if adjacency_matrix[i, j] != 0:
                neighbors[i].append(j)
                weights[i].append(adjacency_matrix[i, j])
    
    # Create PySAL weights object
    w = W(neighbors, weights, silence_warnings=True)
    
    # Calculate bivariate Moran's I
    moran_bv = Moran_BV(x, y, w)
    
    return {
        'I': moran_bv.I,
        'p_value': moran_bv.p_sim,
        'z_score': moran_bv.z_sim
    }


def calculate_adapted_metrics(array1, array2, adjacency_matrix, power=1, normalization='adapted_pearson'):
    """
    Weighted bivariate Moran's I with direct value weighting and normalization.
    Uses w_ij(y_j) = w_ij^spatial * y_j^power (with optional global normalization)
    
    Parameters:
    -----------
    array1 : np.array, shape (n, 1) or (n,)
        First variable (e.g., predictions)
    array2 : np.array, shape (n, 1) or (n,)
        Second variable (e.g., targets, used for weighting)
    adjacency_matrix : np.array, shape (n, n)
        Spatial adjacency/weights matrix
    power : float, default=1
        Weight is array2^power 
        - power=1: linear emphasis on high values
        - power=2: quadratic emphasis (more aggressive)
        - power=0.5: softer emphasis
    normalization : str, default='weighted_variance'
        'weighted_variance': Normalize by weighted variance (guarantees [-1, 1])
        'standard': Keep standard unweighted variance normalization
    
    Formulation:
    ------------
    weighted_variance:
        $$I_{xy} = \frac{\sum_{i}\sum_{j}w_{ij} \cdot y_j^p \cdot z_{x_i} \cdot z_{y_j}}{\sqrt{\sum_{i}\sum_{j}w_{ij} 
        \cdot y_j^p \cdot z_{x_i}^2} \cdot \sqrt{\sum_{i}\sum_{j}w_{ij} \cdot y_j^p \cdot z_{y_j}^2}}$$
        
        where z_{x_i} = x_i - \bar{x}, z_{y_j} = y_j - \bar{y}
        Guarantees I_{xy} ∈ [-1, 1]
    
    standard:
        $$I_{xy} = \frac{n}{\sum_{i}\sum_{j}w_{ij} \cdot y_j^p} \cdot \frac{\sum_{i}\sum_{j}w_{ij} 
        \cdot y_j^p \cdot z_{x_i} \cdot z_{y_j}}{\sqrt{\sum_{i}z_{x_i}^2} \cdot \sqrt{\sum_{i}z_{y_i}^2}}$$
        
        where z_{x_i} = x_i - \bar{x}, z_{y_j} = y_j - \bar{y}
        Similar to standard Moran's I with value-weighted spatial relationships

    """
    # Flatten arrays if needed
    x = array1.flatten()
    y = array2.flatten()
    W = adjacency_matrix.copy().astype(float)
    n = len(x)
    
    # Create value-dependent weights: w_ij(y_j) = w_ij^spatial * y_j^power
    W_weighted = np.zeros_like(W)
    for i in range(n):
        for j in range(n):
            if W[i, j] != 0:
                W_weighted[i, j] = W[i, j] * (y[j] ** power)
            if i == j:  # consider self-connection
                W_weighted[i, j] = 1 * (y[j] ** power)

    # Sum of all weighted connections
    W_sum = np.sum(W_weighted)
    assert W_sum > 0, "Sum of weights must be positive"
    
    # Center the data (using original y values, not normalized)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    
    # Numerator: sum_i sum_j w_ij(y_j) * x_centered_i * y_centered_j
    numerator = 0
    for i in range(n):
        for j in range(n):
            numerator += W_weighted[i, j] * x_centered[i] * y_centered[j]
    
    # Denominator based on normalization method
    if normalization == 'adapted_pearson':
        weighted_var_x = 0
        weighted_var_y = 0
        for i in range(n):
            for j in range(n):
                weighted_var_x += W_weighted[i, j] * x_centered[i]**2
                weighted_var_y += W_weighted[i, j] * y_centered[j]**2
        denominator = np.sqrt(weighted_var_x * weighted_var_y)
        assert denominator != 0, "Denominator in weighted variance normalization cannot be zero"
        I_weighted = numerator / denominator
        
    elif normalization == 'adapted_moran':
        var_x = np.sum(x_centered**2)
        var_y = np.sum(y_centered**2)
        denominator = np.sqrt(var_x * var_y)
        assert denominator != 0, "Denominator in standard normalization cannot be zero"
        I_weighted = (n / W_sum) * (numerator / denominator)
    
    else:
        raise ValueError(f"Unknown normalization method: {normalization}")
    
    return I_weighted


def calculate_weighted_moran_bv_local(array1, array2, adjacency_matrix, power=1, permutations=0):
    """
    Calculate weighted bivariate LISA (Local Indicators of Spatial Association).
    Weights are based on array1 values and normalized to sum to 1.
    Handles isolated locations (nodes with no neighbors).
    
    Parameters:
    -----------
    array1 : np.array, shape (n, 1) or (n,)
        First variable (used for weighting, e.g., predictions)
    array2 : np.array, shape (n, 1) or (n,)
        Second variable (e.g., targets)
    adjacency_matrix : np.array, shape (n, n)
        Spatial adjacency/weights matrix
    power : float, default=1
        Weight emphasis: array1^power (1=linear, 2=quadratic, 0.5=softer)
    permutations : int, default=0
        Number of permutations for significance testing (0 to skip testing)
    
    Returns:
    --------
    dict with:
        - 'weighted_global': Value-weighted global statistic
        - 'local_Is': Array of local statistics for each location
        - 'p_values': Array of p-values for each location (None if permutations=0)
        - 'value_weights': Normalized weights (sum to 1) used in calculation
        - 'n_isolated': Number of isolated locations
    """
    from libpysal.weights import W
    from esda.moran import Moran_Local_BV
    import numpy as np
    
    # Flatten and convert to float64
    x = array1.flatten().astype(np.float64)
    y = array2.flatten().astype(np.float64)
    
    n = len(adjacency_matrix)
    
    # Create normalized value-based weights (sum to 1)
    value_weights = np.abs(x) ** power
    weight_sum = np.sum(value_weights)
    if weight_sum > 1e-10:
        value_weights = value_weights / weight_sum
    else:
        # If all values are zero/near-zero, use uniform weights
        value_weights = np.ones(n) / n
    
    # Build PySAL weights dictionary
    neighbors = {}
    weights_dict = {}
    isolated_count = 0
    
    for i in range(n):
        neighbor_list = []
        weight_list = []
        
        for j in range(n):
            if adjacency_matrix[i, j] != 0:
                neighbor_list.append(j)
                weight_list.append(float(adjacency_matrix[i, j]))
        
        if len(neighbor_list) > 0:
            neighbors[i] = neighbor_list
            weights_dict[i] = weight_list
        else:
            # For isolated locations, add self as neighbor to prevent errors
            neighbors[i] = [i]
            weights_dict[i] = [1.0]
            isolated_count += 1
    
    # Create PySAL weights object
    w = W(neighbors, weights_dict, silence_warnings=True)
    
    # Check for sufficient valid observations
    assert n - isolated_count >= 3
    
    # Calculate local bivariate Moran's I without permutation testing
    lisa_bv = Moran_Local_BV(x, y, w, permutations=permutations)
    
    # Calculate weighted global statistic, use value_weights (normalized to sum to 1)
    weighted_global = float(np.sum(value_weights * lisa_bv.Is))
    
    return {
        'weighted_global': weighted_global,
        'local_Is': lisa_bv.Is,
        'p_values': lisa_bv.p_sim if permutations > 0 else None,
        'value_weights': value_weights,
        'n_isolated': isolated_count
    }


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


def calculate_geo_weighted_correlation(pred, true, adjacency_matrix, bandwidth=2.5, kernel='gaussian', aggr='mean'):
    from scipy.sparse.csgraph import shortest_path

    """
    Calculate Geographically Weighted Correlation for network topology.
    
    Parameters:
    -----------
    pred : np.ndarray
        Predicted values, shape (280, 1)
    true : np.ndarray
        True values, shape (280, 1)
    adjacency_matrix : np.ndarray
        Adjacency matrix defining network topology, shape (280, 280)
        Values are 0 (no connection) or 1 (connected)
    bandwidth : float, default=2.5
        Bandwidth parameter for spatial weighting (in network steps/hops)
        Default of 2.5 means weights decay significantly after 2-3 steps
    kernel : str, default='gaussian'
        Kernel function for spatial weights: 'gaussian' or 'bisquare'
    aggr : str, default='mean'
        Aggregation method: 'mean' or 'weighted_mean'
    
    Returns:
    --------
    gwc : np.ndarray
        Local correlation at each location, shape (280,)
    gwc_aggr : float
        Aggregated correlation value
    """
    # Flatten inputs to 1D arrays
    pred = pred.flatten()
    true = true.flatten()
    
    n = len(pred)
    
    # Calculate shortest path distances (number of steps in network)
    # Replace 0s with inf for non-connected nodes, keep adjacency for connected
    graph = adjacency_matrix.copy().astype(float)
    graph[graph == 0] = np.inf
    np.fill_diagonal(graph, 0)  # Distance to self is 0
    
    # Compute shortest path distance matrix
    distance_matrix = shortest_path(csgraph=graph, directed=False, return_predecessors=False)
    
    # Initialize array for local correlations
    gwc = np.zeros(n)
    
    # Calculate GWC for each location
    for i in range(n):
        # Get distances from location i to all other locations
        distances = distance_matrix[i, :]
        
        # Calculate spatial weights based on kernel function
        if kernel == 'gaussian':
            # Gaussian kernel: w_ij = exp(-d_ij^2 / (2*h^2))
            weights = np.exp(-distances**2 / (2 * bandwidth**2))
        elif kernel == 'bisquare':
            # Bi-square kernel
            weights = np.zeros(n)
            mask = distances < bandwidth
            weights[mask] = (1 - (distances[mask] / bandwidth)**2)**2
        else:
            raise ValueError(f"Unknown kernel: {kernel}. Use 'gaussian' or 'bisquare'")

        # Calculate local weighted means
        sum_weights = np.sum(weights)
        assert sum_weights != 0
            
        X_mean_i = np.sum(weights * pred) / sum_weights
        Y_mean_i = np.sum(weights * true) / sum_weights
        
        # Calculate deviations from local means
        X_dev = pred - X_mean_i
        Y_dev = true - Y_mean_i
        
        # Calculate weighted covariance
        cov_i = np.sum(weights * X_dev * Y_dev)
        
        # Calculate weighted standard deviations
        std_X_i = np.sqrt(np.sum(weights * X_dev**2))
        std_Y_i = np.sqrt(np.sum(weights * Y_dev**2))
        
        # Calculate local correlation
        if std_X_i == 0 or std_Y_i == 0:
            gwc[i] = np.nan
        else:
            gwc[i] = cov_i / (std_X_i * std_Y_i)
    
    # Aggregate the local correlations
    # Remove NaN values for aggregation
    gwc_valid = gwc[~np.isnan(gwc)]
    true_valid = true[~np.isnan(gwc)]
    pred_valid = pred[~np.isnan(gwc)]
    
    if aggr == 'mean':
        gwc_aggr = np.mean(gwc_valid)
    elif aggr == 'weighted_mean':
        gwc_aggr = np.sum(gwc_valid * ((true_valid + pred_valid) / np.sum((true_valid + pred_valid))))
    else:
        raise ValueError(f"Unknown aggregation method: {aggr}. Use 'mean' or 'weighted_mean'")
    
    return gwc_aggr


def cal_metrics(pred, true, label, verbose=0, adj_matrix=None):
    from sklearn.metrics import r2_score
    from scipy.stats import spearmanr, kendalltau, pearsonr

    rmse = np.sqrt(np.mean((pred - true) ** 2))
    mae = np.mean((np.abs(pred - true)))

    non_zero_mask = true != 0
    mape = np.mean(np.abs((true[non_zero_mask] - pred[non_zero_mask]) / (true[non_zero_mask] + 1e-10))) * 100

    w_pearson = calculate_weighted_Pearson(pred, true, p=1)

    geo_pearson = calculate_geo_weighted_correlation(pred, true, adj_matrix, bandwidth=1, aggr='weighted_mean')

    adpated_pearson = calculate_adapted_metrics(
        pred, true, adj_matrix, 
        power=20, normalization='adapted_pearson',
    )

    if verbose > 0:
        print(
            # error metrics
            f'{label} MAE: {mae:.5f}; '
            f'{label} MAPE: {mape:.5f}; '
            # peason
            f'{label} Adapted Pearson: {adpated_pearson:.5f}; ',
            f'{label} Weighted Pearson: {w_pearson:.5f}; '
            f'{label} Geo Pearson Mean: {geo_pearson:.5f};'
        )
    return {
        'MAE': mae, 'MAPE': mape, 'Weighted_Pearson': w_pearson, 'Geo_Pearson': geo_pearson,
        'Adapted_Pearson': adpated_pearson, 
    }


def aggr_metrics(metrics_list_dict, label):
    metric_names = list(metrics_list_dict[0].keys())
    metrics_list = [list(d.values()) for d in metrics_list_dict]
    metrics_array = np.array(metrics_list)
    metrics_mean = np.mean(metrics_array, axis=0)
    print(';\n'.join(f'{label} {n}: {v:.5f}' for n, v in zip(metric_names, metrics_mean)))
    return

