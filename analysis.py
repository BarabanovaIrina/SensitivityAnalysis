import numpy as np
from scipy.stats import norm


def compute_mu_star_conf(ee, conf_level, resamples, traj_num, seed=None):
    np.random.seed(seed)

    if not 0 < conf_level < 1:
        raise ValueError("Confidence level must be between 0-1.")

    resample_index = np.random.randint(len(ee), size=(resamples, traj_num))
    ee_resampled = ee[resample_index]
    # Compute average of the absolute values over each of the resamples
    mu_star_resampled = np.average(np.abs(ee_resampled), axis=1)

    return norm.ppf(0.5 + conf_level / 2) * mu_star_resampled.std(ddof=1)


def get_increased_values(op_vec, up, lo):
    up = np.pad(up, ((0, 0), (1, 0), (0, 0)), 'constant')
    lo = np.pad(lo, ((0, 0), (0, 1), (0, 0)), 'constant')

    res = np.einsum('ik,ikj->ij', op_vec, up + lo)

    return res.T


def get_decreased_values(op_vec, up, lo):
    up = np.pad(up, ((0, 0), (0, 1), (0, 0)), 'constant')
    lo = np.pad(lo, ((0, 0), (1, 0), (0, 0)), 'constant')

    res = np.einsum('ik,ikj->ij', op_vec, up + lo)

    return res.T


def evaluate_elem_effect(model_input, model_output, traj_num, delta):
    traj_size = int(model_output.size / traj_num)
    num_rows = model_input.shape[0]
    elem_eff = np.zeros((traj_num, num_rows), dtype=np.float)
    vec = model_input.reshape(traj_num, traj_size, model_input.shape[1])
    cha = np.subtract(vec[:, 1:, :], vec[:, 0:-1, :])
    up = (cha > 0)
    lo = (cha < 0)

    op_vec = model_output.reshape(traj_num, traj_size)

    result_up = get_increased_values(op_vec, up, lo)
    result_lo = get_decreased_values(op_vec, up, lo)

    elem_eff = np.subtract(result_up, result_lo)
    np.divide(elem_eff, delta, out=elem_eff)

    return elem_eff


def morris(input_params, sample_input, sample_output, resamples=1000, num_levels=4, conf_level=0.95, seed=None):
    if seed:
        np.random.seed(seed)

    sample_input.astype('float64')
    sample_output.astype('float64')

    params_number = input_params.size
    delta = num_levels / (2.0 * (num_levels - 1))

    num_trajectories = int(sample_output.size / (params_number + 1))
    elementary_effect = evaluate_elem_effect(sample_input,
                                             sample_output,
                                             num_trajectories,
                                             delta)

    si = dict()
    si['names'] = input_params.names
    si['mu'] = np.average(elementary_effect, 1)
    si['mu_star'] = np.average(np.abs(elementary_effect), 1)
    si['sigma'] = np.std(elementary_effect, axis=1, ddof=1)
    si['mu_star_conf'] = list()

    for i in range(params_number):
        i_star = compute_mu_star_conf(elementary_effect[i, :],
                                      conf_level,
                                      resamples,
                                      num_trajectories)

        si['mu_star_conf'].append(i_star)

    return si


print('necessary for commit')