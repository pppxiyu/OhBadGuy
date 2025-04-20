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


def cal_metrics(pred, true, label, verbose=0):
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    mae = np.mean((np.abs(pred - true)))
    kl_d = kl_divergence(pred, true)
    js_d = js_divergence(pred, true)
    if verbose > 0:
        print(
            f'{label} RMSE: {rmse:.5f}; '
            f'{label} MAE: {mae:.5f}; '
            f'{label} KL-Divergence: {kl_d:.5f}; '
            f'{label} JS-Divergence: {js_d:.5f}'
        )
    return {
        'RMSE': rmse, 'MAE': mae, 'KL-Divergence': kl_d, 'JS_Divergence': js_d
    }


def aggr_metrics(metrics_list_dict, label):
    metric_names = list(metrics_list_dict[0].keys())
    metrics_list = [list(d.values()) for d in metrics_list_dict]
    metrics_array = np.array(metrics_list)
    metrics_mean = np.mean(metrics_array, axis=0)
    print(';\n'.join(f'{label} {n}: {v:.5f}' for n, v in zip(metric_names, metrics_mean)))
    return

