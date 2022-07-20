import numpy as np
import torch

from gpytorch.utils.permutation import apply_permutation

pp = lambda x: np.array2string(x, precision=4, floatmode="fixed")


def pivoted_cholesky_np_gpt(mat: np.matrix, error_tol=1e-6, max_iter=np.Infinity):
    """
    mat: numpy matrix of N x N

    This is to replicate what is done in GPyTorch verbatim.
    """
    n = mat.shape[-1]
    max_iter = min(int(max_iter), n)

    d = np.array(np.diag(mat))
    orig_error = np.max(d)
    error = np.linalg.norm(d, 1) / orig_error
    pi = np.arange(n)

    L = np.zeros((max_iter, n))

    m = 0
    while m < max_iter and error > error_tol:
        permuted_d = d[pi]
        max_diag_idx = np.argmax(permuted_d[m:])
        max_diag_idx = max_diag_idx + m
        max_diag_val = permuted_d[max_diag_idx]
        i = max_diag_idx

        # swap pi_m and pi_i
        pi[m], pi[i] = pi[i], pi[m]
        pim = pi[m]

        L[m, pim] = np.sqrt(max_diag_val)

        if m + 1 < n:
            row = apply_permutation(
                torch.from_numpy(mat), torch.tensor(pim), right_permutation=None
            )  # left permutation just swaps row
            row = row.numpy().flatten()
            pi_i = pi[m + 1 :]
            L_m_new = row[pi_i]  # length = 9

            if m > 0:
                L_prev = L[:m, pi_i]
                update = L[:m, pim]
                prod = update @ L_prev
                L_m_new = L_m_new - prod  # np.sum(prod, axis=-1)

            L_m = L[m, :]
            L_m_new = L_m_new / L_m[pim]
            L_m[pi_i] = L_m_new

            matrix_diag_current = d[pi_i]
            d[pi_i] = matrix_diag_current - L_m_new**2

            L[m, :] = L_m
            error = np.linalg.norm(d[pi_i], 1) / orig_error
        m = m + 1
    return L, pi
