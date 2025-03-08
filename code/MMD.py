import torch

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5):
    """
    Computes a multi-Gaussian kernel between source and target features.

    Args:
        source (torch.Tensor): Source domain features, shape [n_s, d].
        target (torch.Tensor): Target domain features, shape [n_t, d].
        kernel_mul (float, optional): Multiplicative factor for kernel widths. Defaults to 2.0.
        kernel_num (int, optional): Number of kernels. Defaults to 5.

    Returns:
        torch.Tensor: Combined kernel matrix, shape [n, n], where n = n_s + n_t.
    """
    n_s = source.size(0)
    n_t = target.size(0)
    n = n_s + n_t

    # Concatenate source and target features
    total = torch.cat([source, target], dim=0)  # Shape: [n, d]

    # Compute pairwise squared L2 distances
    # Using (x - y)^2 = x^2 + y^2 - 2xy
    total_square = torch.sum(total ** 2, dim=1, keepdim=True)  # Shape: [n, 1]
    L2_distance = total_square + total_square.t() - 2 * torch.matmul(total, total.t())  # Shape: [n, n]
    L2_distance = torch.clamp(L2_distance, min=0.0)  # Ensure numerical stability

    # Compute the length scale (sigma)
    # Equivalent to: sum(L2_distance) / (n^2 - n)
    length_scale = L2_distance.sum() / (n**2 - n)

    # Adjust the length scale based on kernel multiplication factor
    length_scale /= kernel_mul ** (kernel_num // 2)

    # Generate a list of length scales for multiple kernels
    length_scale_list = [length_scale * (kernel_mul ** i) for i in range(kernel_num)]

    # Compute multi-Gaussian kernels and sum them up
    M_kernel = [torch.exp(-L2_distance / scale) for scale in length_scale_list]  # List of [n, n] tensors
    M_kernel = torch.stack(M_kernel, dim=0)  # Shape: [kernel_num, n, n]
    M_kernel = torch.sum(M_kernel, dim=0)  # Shape: [n, n]

    return M_kernel

def MK_MMD(source, target, kernel_mul=2.0, kernel_num=5):
    """
    Computes the Multiple Kernel Maximum Mean Discrepancy (MK-MMD) loss between source and target.

    Args:
        source (torch.Tensor): Source domain features, shape [n_s, d].
        target (torch.Tensor): Target domain features, shape [n_t, d].
        kernel_mul (float, optional): Multiplicative factor for kernel widths. Defaults to 2.0.
        kernel_num (int, optional): Number of kernels. Defaults to 5.

    Returns:
        torch.Tensor: MK-MMD loss (scalar).
    """
    # Compute the combined kernel matrix
    kernels = gaussian_kernel(source, target, kernel_mul=kernel_mul, kernel_num=kernel_num)  # Shape: [n, n]

    n_s = source.size(0)
    n_t = target.size(0)

    # Extract sub-kernels for different domain pairs
    XX = kernels[:n_s, :n_s].sum() / (n_s ** 2)          # Source vs Source
    YY = kernels[n_s:, n_s:].sum() / (n_t ** 2)          # Target vs Target
    XY = kernels[:n_s, n_s:].sum() / (n_s * n_t)        # Source vs Target
    YX = kernels[n_s:, :n_s].sum() / (n_s * n_t)        # Target vs Source

    # Compute MK-MMD loss
    loss = torch.abs(XX + YY - XY - YX)

    return loss

def compute_js_divergence(p, q):
    """
    计算Jensen-Shannon (JS) Divergence：DJS(p || q) = 0.5 * (DKL(p || m) + DKL(q || m))
    其中 m = 0.5 * (p + q)
    """

    m = 0.5 * (p + q)
    
    
    kl_pm = compute_kl_divergence(p, m)
    kl_qm = compute_kl_divergence(q, m)

    # 计算JS散度
    js_divergence = 0.5 * (kl_pm + kl_qm)
    return js_divergence

def compute_kl_divergence(p, m):
   
    kl = torch.sum(p * torch.log(p / m), dim=1).mean()  
    return kl