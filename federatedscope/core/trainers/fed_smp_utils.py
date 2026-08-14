"""
Fed-SMP: Federated Learning with Sparsified Model Perturbation

工具函数：噪声乘数计算、mask 生成、梯度加噪

References:
    Hu et al., "Federated Learning with Sparsified Model Perturbation:
    Improving Accuracy under Client-Level Differential Privacy", 2022.
    https://arxiv.org/abs/2202.07178
"""

import math
import logging
import torch

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# 1. 噪声乘数计算
# ------------------------------------------------------------------ #

def compute_sigma(epsilon, delta, total_rounds, sample_rate):
    """
    由隐私预算 (ε, δ) 推导噪声乘数 σ（Fed-SMP Theorem 1 简化公式）

    满足 (ε, δ)-DP 的充分条件：
        σ² ≥ 7 * q² * T * (ε + 2·ln(1/δ)) / ε²

    注：此公式为充分条件，给出的 σ 偏保守。
        若安装了 Opacus，建议改用 compute_sigma_opacus() 获得更紧的界。

    Args:
        epsilon:      隐私预算 ε（越小隐私保护越强）
        delta:        隐私参数 δ，通常设为 1/n^1.1
        total_rounds: 总通信轮数 T
        sample_rate:  客户端采样率 q = r/n

    Returns:
        sigma: 噪声乘数 σ
    """
    numerator = 7.0 * (sample_rate ** 2) * total_rounds * (
        epsilon + 2.0 * math.log(1.0 / delta)
    )
    sigma = math.sqrt(numerator) / epsilon
    logger.info(
        f"[Fed-SMP] compute_sigma: ε={epsilon}, δ={delta}, "
        f"T={total_rounds}, q={sample_rate:.4f} → σ={sigma:.4f}"
    )
    return sigma


def compute_sigma_opacus(epsilon, delta, total_rounds, sample_rate):
    """
    使用 Opacus RDP accountant 计算噪声乘数（结果比 Theorem 1 更紧）

    需要安装 Opacus：pip install opacus

    Args:
        同 compute_sigma()

    Returns:
        sigma: 噪声乘数 σ
    """
    try:
        from opacus.accountants.utils import get_noise_multiplier
        sigma = get_noise_multiplier(
            target_epsilon=epsilon,
            target_delta=delta,
            sample_rate=sample_rate,
            epochs=total_rounds,
        )
        logger.info(
            f"[Fed-SMP] compute_sigma_opacus: ε={epsilon}, δ={delta}, "
            f"T={total_rounds}, q={sample_rate:.4f} → σ={sigma:.4f}"
        )
        return sigma
    except ImportError:
        logger.warning(
            "[Fed-SMP] Opacus not installed, falling back to Theorem 1 formula."
        )
        return compute_sigma(epsilon, delta, total_rounds, sample_rate)


# ------------------------------------------------------------------ #
# 2. Mask 生成（服务器端调用）
# ------------------------------------------------------------------ #

def generate_randk_mask(model_state_dict, compress_ratio):
    """
    生成随机稀疏 mask（rand_k sparsifier）

    对每个参数张量，随机选取 k = compress_ratio * d 个坐标置 True，
    其余置 False。同一轮所有客户端使用同一个 mask。

    Args:
        model_state_dict: 全局模型 state_dict，用于确定各层形状
        compress_ratio:   压缩率 p = k/d，取值 (0, 1]

    Returns:
        mask: dict {param_name: BoolTensor}，True 表示该坐标被保留
    """
    mask = {}
    for key, param in model_state_dict.items():
        if not torch.is_tensor(param):
            continue
        d = param.numel()
        k = max(1, int(compress_ratio * d))
        perm = torch.randperm(d)
        m = torch.zeros(d, dtype=torch.bool)
        m[perm[:k]] = True
        mask[key] = m.view(param.shape)

    total_d = sum(p.numel() for p in model_state_dict.values())
    total_k = sum(m.sum().item() for m in mask.values())
    logger.info(
        f"[Fed-SMP] rand_k mask: keep {int(total_k)}/{total_d} "
        f"({100.0 * total_k / total_d:.2f}%) coordinates"
    )
    return mask


def generate_topk_mask(model, public_dataloader, compress_ratio,
                       device='cpu', local_steps=1):
    """
    生成 top-k mask（top_k sparsifier）

    服务器在公开数据集上跑 local_steps 步 SGD，
    选取模型更新量绝对值最大的 k 个坐标。

    Args:
        model:             全局模型
        public_dataloader: 公开数据集 DataLoader（分布与客户端数据相近）
        compress_ratio:    压缩率 p = k/d
        device:            运算设备
        local_steps:       SGD 步数（论文默认与客户端本地步数相同）

    Returns:
        mask: dict {param_name: BoolTensor}
    """
    import copy
    model_copy = copy.deepcopy(model).to(device)
    model_copy.train()
    optimizer = torch.optim.SGD(model_copy.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()

    theta_init = {k: v.clone() for k, v in model_copy.state_dict().items()}

    step = 0
    for batch in public_dataloader:
        if step >= local_steps:
            break
        x, y = batch[0].to(device), batch[1].to(device)
        optimizer.zero_grad()
        loss = criterion(model_copy(x), y)
        loss.backward()
        optimizer.step()
        step += 1

    theta_end = model_copy.state_dict()

    mask = {}
    for key in theta_init:
        if key not in theta_end:
            continue
        delta_abs = (theta_init[key] - theta_end[key]).abs().view(-1)
        d = delta_abs.numel()
        k = max(1, int(compress_ratio * d))
        topk_indices = torch.topk(delta_abs, k).indices
        m = torch.zeros(d, dtype=torch.bool)
        m[topk_indices] = True
        mask[key] = m.view(theta_init[key].shape)

    total_d = sum(p.numel() for p in theta_init.values())
    total_k = sum(m.sum().item() for m in mask.values())
    logger.info(
        f"[Fed-SMP] top_k mask: keep {int(total_k)}/{total_d} "
        f"({100.0 * total_k / total_d:.2f}%) coordinates"
    )
    return mask


# ------------------------------------------------------------------ #
# 3. 梯度加噪（客户端调用）
# ------------------------------------------------------------------ #

def apply_fed_smp_noise(gradients, mask, clip_norm, sigma, sample_client_num):
    """
    对梯度施加 Fed-SMP 差分隐私处理：选中子向量裁剪/加噪，未选中坐标原样上传

    当前实现按实验需求调整为：
        1. 取被 mask 选中的子向量 g_sel = g ⊙ m
        2. 仅对 g_sel 做全局 L2 裁剪
        3. 仅对选中坐标加高斯噪声
        4. 未选中坐标保持原始梯度不变并正常上传

    每个被选中坐标的噪声标准差 = C * σ / √r

    Args:
        gradients:         dict {param_name: Tensor}，原始梯度
        mask:              dict {param_name: BoolTensor}，服务器广播的 mask
        clip_norm:         L2 裁剪阈值 C
        sigma:             噪声乘数 σ
        sample_client_num: 每轮参与的客户端数 r

    Returns:
        noisy_gradients: dict {param_name: Tensor}，仅选中坐标被扰动后的梯度
    """
    # 噪声标准差 = C * σ / √r （论文 Algorithm 2 line 39）
    noise_std = clip_norm * sigma / math.sqrt(sample_client_num)

    # step 1: 提取被选中的子向量；未选中坐标后续保持原值
    selected_grads = {}
    for key, grad in gradients.items():
        if not torch.is_tensor(grad):
            continue
        if not torch.is_floating_point(grad):
            selected_grads[key] = grad.clone()
            continue
        if mask is not None and key in mask:
            m = mask[key].to(grad.device).float()
            selected_grads[key] = grad * m
        else:
            selected_grads[key] = grad.clone()

    # step 2: 仅对选中子向量做全局 L2 裁剪
    total_norm = math.sqrt(
        sum(g.norm(2).item() ** 2 for g in selected_grads.values()
            if torch.is_floating_point(g))
    )
    if total_norm > clip_norm:
        clip_factor = clip_norm / (total_norm + 1e-10)
        for key in selected_grads:
            if not torch.is_floating_point(selected_grads[key]):
                continue
            selected_grads[key] = selected_grads[key] * clip_factor

    # step 3: 只在被选中的坐标上加高斯噪声，未选中坐标保持原始值
    noisy_gradients = {}
    total_abs_perturb_all = 0.0
    total_coords_all = 0
    total_abs_perturb_kept = 0.0
    total_coords_kept = 0
    for key, grad in gradients.items():
        if not torch.is_tensor(grad):
            continue
        if not torch.is_floating_point(grad):
            noisy_gradients[key] = grad.clone()
            continue

        protected_grad = selected_grads[key]
        output_grad = grad.clone()
        noise = torch.zeros_like(grad)

        if mask is not None and key in mask:
            kept_mask = mask[key].to(grad.device).bool()
        else:
            kept_mask = torch.ones_like(grad, dtype=torch.bool)

        kept_num = int(kept_mask.sum().item())
        if kept_num > 0:
            noise = torch.normal(
                mean=0.0, std=noise_std,
                size=grad.shape, device=grad.device
            )
            noise = noise * kept_mask.float()
            output_grad[kept_mask] = protected_grad[kept_mask] + noise[kept_mask]
            total_abs_perturb_kept += noise.abs()[kept_mask].sum().item()
            total_coords_kept += kept_num

        total_abs_perturb_all += noise.abs().sum().item()
        total_coords_all += noise.numel()
        noisy_gradients[key] = output_grad

    mean_abs_perturb_all = total_abs_perturb_all / max(total_coords_all, 1)
    mean_abs_perturb_kept = total_abs_perturb_kept / max(total_coords_kept, 1)

    logger.info(
        f"[Fed-SMP] apply_noise: C={clip_norm}, σ={sigma:.4f}, "
        f"r={sample_client_num}, noise_std={noise_std:.6f}, "
        f"selected_grad_norm_before={total_norm:.4f}, "
        f"mean_abs_perturb_all={mean_abs_perturb_all:.6f}, "
        f"mean_abs_perturb_kept={mean_abs_perturb_kept:.6f}"
    )
    return noisy_gradients
