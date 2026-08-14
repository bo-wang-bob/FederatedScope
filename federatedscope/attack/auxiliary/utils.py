import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import logging
import os
import numpy as np
import federatedscope.register as register
try:
    from skimage.metrics import structural_similarity as ssim
except Exception:  # pragma: no cover
    ssim = None


logger = logging.getLogger(__name__)


def _resize_like_if_needed(reference, candidate):
    reference = np.asarray(reference)
    candidate = np.asarray(candidate)
    if reference.shape == candidate.shape:
        return reference, candidate

    if reference.ndim >= 2 and candidate.ndim >= 2 and \
            reference.shape[:2] != candidate.shape[:2]:
        cand = torch.from_numpy(candidate).float()
        if candidate.ndim == 2:
            cand = cand.unsqueeze(0).unsqueeze(0)
        elif candidate.ndim == 3:
            cand = cand.permute(2, 0, 1).unsqueeze(0)
        else:
            return reference, candidate
        cand = F.interpolate(
            cand, size=reference.shape[:2], mode='bilinear',
            align_corners=False)
        if candidate.ndim == 2:
            candidate = cand.squeeze(0).squeeze(0).numpy()
        else:
            candidate = cand.squeeze(0).permute(1, 2, 0).numpy()

    return reference, candidate


def _fallback_ssim(original, reconstructed, data_range=1.0):
    original = np.asarray(original, dtype=np.float64)
    reconstructed = np.asarray(reconstructed, dtype=np.float64)
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    if original.ndim == 2:
        channels = [(original, reconstructed)]
    else:
        channels = [
            (original[..., c], reconstructed[..., c])
            for c in range(original.shape[-1])
        ]

    values = []
    for orig_c, recon_c in channels:
        mux = orig_c.mean()
        muy = recon_c.mean()
        varx = ((orig_c - mux) ** 2).mean()
        vary = ((recon_c - muy) ** 2).mean()
        cov = ((orig_c - mux) * (recon_c - muy)).mean()
        numerator = (2 * mux * muy + c1) * (2 * cov + c2)
        denominator = (mux ** 2 + muy ** 2 + c1) * (varx + vary + c2)
        values.append(numerator / denominator if denominator != 0 else 1.0)
    return float(np.mean(values))


def calculate_ssim(original, reconstructed, data_range=1.0, multichannel=True):
    try:
        if torch.is_tensor(original):
            original = original.detach().cpu().numpy()
        if torch.is_tensor(reconstructed):
            reconstructed = reconstructed.detach().cpu().numpy()

        original, reconstructed = _resize_like_if_needed(
            original, reconstructed)

        if ssim is None:
            return _fallback_ssim(original, reconstructed, data_range)

        if len(original.shape) == 2:
            return float(ssim(original, reconstructed, data_range=data_range))

        try:
            return float(ssim(original, reconstructed,
                              data_range=data_range,
                              channel_axis=2))
        except TypeError:
            return float(ssim(original, reconstructed,
                              data_range=data_range,
                              multichannel=True))
    except Exception as e:
        logger.warning(f"SSIM calculation failed: {e}")
        return _fallback_ssim(original, reconstructed, data_range)


def calculate_psnr(original, reconstructed, data_range=1.0):
    try:
        if torch.is_tensor(original):
            original = original.cpu().numpy()
        if torch.is_tensor(reconstructed):
            reconstructed = reconstructed.cpu().numpy()
        original, reconstructed = _resize_like_if_needed(
            original, reconstructed)
        mse = np.mean((original - reconstructed) ** 2)
        if mse == 0:
            return float('inf')
        return float(10 * np.log10((data_range ** 2) / mse))
    except Exception as e:
        logger.warning(f"PSNR calculation failed: {e}")
        return -1.0


def wasserstein_distance(x, y, device='cpu', sample_size=100000):
    x_flat = x.flatten()
    y_flat = y.flatten()
    total_size = x_flat.shape[0]
    if total_size > sample_size:
        indices = torch.randperm(total_size, device=device)[:sample_size]
        x_flat = x_flat[indices]
        y_flat = y_flat[indices]
    x_sorted, _ = torch.sort(x_flat)
    y_sorted, _ = torch.sort(y_flat)
    return torch.mean(torch.abs(x_sorted - y_sorted))


def label_to_onehot(target, num_classes=100):
    return torch.nn.functional.one_hot(target, num_classes)


def cross_entropy_for_onehot(pred, target):
    return torch.mean(torch.sum(-target * F.log_softmax(pred, dim=-1), 1))


def iDLG_trick(original_gradient, num_class, is_one_hot_label=False):
    '''
    Using iDLG trick to recover the label. Paper: "iDLG: Improved Deep
    Leakage from Gradients", link: https://arxiv.org/abs/2001.02610

    Args:
        original_gradient: the gradient of the FL model; type: list
        num_class: the total number of class in the data
        is_one_hot_label: whether the dataset's label is in the form of one
        hot. Type: bool

    Returns:
        The recovered label by iDLG trick.

    '''
    last_weight_min = torch.argmin(torch.sum(original_gradient[-2], dim=-1),
                                   dim=-1).detach()

    if is_one_hot_label:
        label = label_to_onehot(
            last_weight_min.reshape((1, )).requires_grad_(False), num_class)
    else:
        label = last_weight_min
    return label


def cos_sim(input_gradient, gt_gradient):
    total = 1 - torch.nn.functional.cosine_similarity(
        input_gradient.flatten(), gt_gradient.flatten(), 0, 1e-10)

    # total = 0
    # input_norm= 0
    # gt_norm = 0
    #
    # total -= (input_gradient * gt_gradient).sum()
    # input_norm += input_gradient.pow(2).sum()
    # gt_norm += gt_gradient.pow(2).sum()
    # total += 1 + total / input_norm.sqrt() / gt_norm.sqrt()

    return total


def total_variation(x):
    """Anisotropic TV."""
    dx = torch.mean(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
    dy = torch.mean(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))

    total = x.size()[0]
    for ind in range(1, len(x.size())):
        total *= x.size()[ind]
    return (dx + dy) / (total)


def approximate_func(x, device, C1=20, C2=0.5):
    '''
    Approximate the function f(x) = 0 if x<0.5 otherwise 1
    Args:
        x: input data;
        device:
        C1:
        C2:

    Returns:
        1/(1+e^{-1*C1 (x-C2)})

    '''
    C1 = torch.tensor(C1).to(torch.device(device))
    C2 = torch.tensor(C2).to(torch.device(device))

    return 1 / (1 + torch.exp(-1 * C1 * (x - C2)))


def get_classifier(classifier: str, model=None):
    if model is not None:
        return model

    if classifier == 'lr':
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(random_state=0)
        return model
    elif classifier.lower() == 'randomforest':
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(random_state=0)
        return model
    elif classifier.lower() == 'svm':
        from sklearn.svm import SVC
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import make_pipeline
        model = make_pipeline(StandardScaler(), SVC(gamma='auto'))
        return model
    else:
        ValueError()


def get_data_info(dataset_name):
    '''
    Get the dataset information, including the feature dimension, number of
    total classes, whether the label is represented in one-hot version

    Args:
        dataset_name:dataset name; str

    :returns:
        data_feature_dim, num_class, is_one_hot_label

    '''
    if dataset_name.lower() == 'femnist':
        return [1, 28, 28], 36, False
    elif 'mnist' in dataset_name.lower():
        return [1, 28, 28], 10, False
    elif 'cifar10' in dataset_name.lower():
        return [3, 32, 32], 10, False
    elif 'office' in dataset_name.lower() and 'home' in dataset_name.lower():
        return [3, 128, 128], 65, False
    else:
        raise ValueError(
            'Please provide the data info of {}: data_feature_dim, num_class'.
            format(dataset_name))


def get_data_sav_fn(dataset_name):
    if dataset_name.lower() == 'femnist':
        return sav_femnist_image
    elif 'mnist' in dataset_name.lower():
        return sav_mnist_image
    elif 'cifar10' in dataset_name.lower():
        return sav_cifar10_image
    elif 'office' in dataset_name.lower() and 'home' in dataset_name.lower():
        return sav_officehome_image  # 使用专门的 OfficeHome 保存函数（ImageNet 归一化）
    else:
        logger.info(f"Reconstructed data saving function is not provided for "
                    f"dataset: {dataset_name}")
        return None


def _postprocess_mnist_recon(img):
    img = torch.clamp(img, 0, 1).clone()

    # Estimate the dark background from lower quantiles, then stretch contrast.
    flat = img.flatten()
    lo = torch.quantile(flat, 0.60)
    hi = torch.quantile(flat, 0.995)
    if hi > lo:
        img = (img - lo) / (hi - lo)
    else:
        img = img - lo
    img = torch.clamp(img, 0, 1)

    # Suppress residual gray haze while keeping thin strokes.
    threshold = min(0.35, max(0.12, float(img.mean() + 0.25 * img.std())))
    img[img < threshold] = 0.0

    max_val = img.max()
    if max_val > 0:
        img = img / max_val
    return img


def sav_mnist_image(data, sav_pth, name, original_data=None):
    """Save MNIST reconstructed images with black-background post-processing."""
    if len(data.shape) == 2:
        data = torch.unsqueeze(data, 0)
        data = torch.unsqueeze(data, 0)

    ind = min(data.shape[0], 16)

    if original_data is not None:
        if len(original_data.shape) == 2:
            original_data = torch.unsqueeze(original_data, 0)
            original_data = torch.unsqueeze(original_data, 0)

        ssim_values = []
        psnr_values = []
        for i in range(min(ind, 8)):
            orig_img = original_data[i, 0, :, :].cpu()
            orig_img = orig_img * 0.1592 + 0.9637
            orig_img = torch.clamp(orig_img, 0, 1).numpy()

            recon_img = _postprocess_mnist_recon(data[i, 0, :, :].cpu()).numpy()
            ssim_values.append(calculate_ssim(orig_img, recon_img, data_range=1.0, multichannel=False))
            psnr_values.append(calculate_psnr(orig_img, recon_img, data_range=1.0))

        avg_ssim = np.mean(ssim_values) if ssim_values else -1.0
        avg_psnr = np.mean(psnr_values) if psnr_values else -1.0
        logger.info(f"[METRICS] MNIST reconstruction quality (black-background postprocessed) - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f} dB | Individual SSIM: {[f'{s:.3f}' for s in ssim_values]}, PSNR: {[f'{p:.2f}' for p in psnr_values]}")

        fig = plt.figure(figsize=(8, 4))
        rows, cols = 2, min(ind, 8)
        for i in range(min(ind, 8)):
            plt.subplot(rows, cols, i + 1)
            orig_img = original_data[i, 0, :, :].cpu()
            orig_img = orig_img * 0.1592 + 0.9637
            orig_img = torch.clamp(orig_img * 255, 0, 255)
            plt.imshow(orig_img, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            if i == 0:
                plt.title(f"Original (SSIM: {avg_ssim:.3f}, PSNR: {avg_psnr:.1f}dB)", fontsize=8)

            plt.subplot(rows, cols, cols + i + 1)
            recon_img = _postprocess_mnist_recon(data[i, 0, :, :].cpu()) * 255
            plt.imshow(recon_img, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            if i == 0:
                plt.title(f"Reconstructed (SSIM: {ssim_values[i]:.3f}, PSNR: {psnr_values[i]:.1f}dB)", fontsize=8)
    else:
        fig = plt.figure(figsize=(4, 4))
        for i in range(ind):
            plt.subplot(4, 4, i + 1)
            recon_img = _postprocess_mnist_recon(data[i, 0, :, :].cpu()) * 255
            plt.imshow(recon_img, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(sav_pth, name))
    plt.close()

def sav_femnist_image(data, sav_pth, name, original_data=None):
    """Save FEMNIST reconstructed images with optional original comparison

    Args:
        data: Reconstructed images [B, 1, H, W]
        sav_pth: Save path
        name: Filename
        original_data: Optional original images for comparison [B, 1, H, W]

    Note: GRNN generator outputs images in [0,1] range (sigmoid activation).
    We scale to [0, 255] for grayscale display.
    """
    if len(data.shape) == 2:
        data = torch.unsqueeze(data, 0)
        data = torch.unsqueeze(data, 0)

    ind = min(data.shape[0], 16)

    # Determine grid size based on whether we have original data
    if original_data is not None:
        # Show original and reconstructed side by side
        if len(original_data.shape) == 2:
            original_data = torch.unsqueeze(original_data, 0)
            original_data = torch.unsqueeze(original_data, 0)

        # 计算SSIM和PSNR值
        ssim_values = []
        psnr_values = []
        for i in range(min(ind, 8)):
            # 准备原始图像（反归一化）
            orig_img = original_data[i, 0, :, :].cpu()
            orig_img = orig_img * 0.1592 + 0.9637
            orig_img = torch.clamp(orig_img, 0, 1).numpy()

            # 准备重建图像
            recon_img = data[i, 0, :, :].cpu()
            recon_img = torch.clamp(recon_img, 0, 1).numpy()

            # 计算SSIM
            ssim_val = calculate_ssim(orig_img, recon_img, data_range=1.0, multichannel=False)
            ssim_values.append(ssim_val)

            # 计算PSNR
            psnr_val = calculate_psnr(orig_img, recon_img, data_range=1.0)
            psnr_values.append(psnr_val)

        # 计算平均SSIM和PSNR
        avg_ssim = np.mean(ssim_values) if ssim_values else -1.0
        avg_psnr = np.mean(psnr_values) if psnr_values else -1.0
        logger.info(f"[METRICS] FEMNIST reconstruction quality - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f} dB | Individual SSIM: {[f'{s:.3f}' for s in ssim_values]}, PSNR: {[f'{p:.2f}' for p in psnr_values]}")

        fig = plt.figure(figsize=(8, 4))
        rows, cols = 2, min(ind, 8)  # 2 rows: original + reconstructed

        for i in range(min(ind, 8)):
            # Original image (top row)
            plt.subplot(rows, cols, i + 1)
            orig_img = original_data[i, 0, :, :].cpu()
            # Denormalize original (FEMNIST uses mean=0.9637, std=0.1592)
            orig_img = orig_img * 0.1592 + 0.9637
            orig_img = torch.clamp(orig_img * 255, 0, 255)
            plt.imshow(orig_img, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            if i == 0:
                plt.title(f'Original (SSIM: {avg_ssim:.3f}, PSNR: {avg_psnr:.1f}dB)', fontsize=8)

            # Reconstructed image (bottom row)
            plt.subplot(rows, cols, cols + i + 1)
            recon_img = data[i, 0, :, :] * 255
            plt.imshow(recon_img, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')
            if i == 0:
                plt.title(f'Reconstructed (SSIM: {ssim_values[i]:.3f}, PSNR: {psnr_values[i]:.1f}dB)', fontsize=8)
    else:
        # Only show reconstructed images
        fig = plt.figure(figsize=(4, 4))
        for i in range(ind):
            plt.subplot(4, 4, i + 1)
            plt.imshow(data[i, 0, :, :] * 255, cmap='gray', vmin=0, vmax=255)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(sav_pth, name))
    plt.close()


def sav_cifar10_image(data, sav_pth, name, original_data=None):
    """Save CIFAR10 reconstructed images with optional original comparison

    Args:
        data: Reconstructed images [B, C, H, W]
        sav_pth: Save path
        name: Filename
        original_data: Optional original images for comparison [B, C, H, W]

    Note: GRNN generator outputs images in [0,1] range (sigmoid activation),
    so we don't need to denormalize. We display them directly.
    """
    if len(data.shape) == 3:  # single image [C, H, W]
        data = torch.unsqueeze(data, 0)

    ind = min(data.shape[0], 16)

    # Determine grid size based on whether we have original data
    if original_data is not None:
        # Show original and reconstructed side by side
        if len(original_data.shape) == 3:
            original_data = torch.unsqueeze(original_data, 0)

        # 计算SSIM和PSNR值
        ssim_values = []
        psnr_values = []
        for i in range(min(ind, 8)):
            # 准备原始图像（反归一化）
            orig_img = original_data[i].cpu()
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
            orig_img = orig_img * std + mean
            orig_img = torch.clamp(orig_img, 0, 1)
            orig_img = orig_img.permute(1, 2, 0).numpy()

            # 准备重建图像
            recon_img = data[i].cpu()
            recon_img = torch.clamp(recon_img, 0, 1)
            recon_img = recon_img.permute(1, 2, 0).numpy()

            # 计算SSIM
            ssim_val = calculate_ssim(orig_img, recon_img, data_range=1.0, multichannel=True)
            ssim_values.append(ssim_val)

            # 计算PSNR
            psnr_val = calculate_psnr(orig_img, recon_img, data_range=1.0)
            psnr_values.append(psnr_val)

        # 计算平均SSIM和PSNR
        avg_ssim = np.mean(ssim_values) if ssim_values else -1.0
        avg_psnr = np.mean(psnr_values) if psnr_values else -1.0
        logger.info(f"[METRICS] CIFAR-10 reconstruction quality - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f} dB | Individual SSIM: {[f'{s:.3f}' for s in ssim_values]}, PSNR: {[f'{p:.2f}' for p in psnr_values]}")

        fig = plt.figure(figsize=(8, 4))
        rows, cols = 2, min(ind, 8)  # 2 rows: original + reconstructed

        for i in range(min(ind, 8)):
            # Original image (top row)
            plt.subplot(rows, cols, i + 1)
            orig_img = original_data[i].cpu()
            # Denormalize original (CIFAR-10 normalization)
            mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616]).view(3, 1, 1)
            orig_img = orig_img * std + mean
            orig_img = torch.clamp(orig_img, 0, 1)
            orig_img = orig_img.permute(1, 2, 0).numpy()
            plt.imshow(orig_img)
            plt.axis('off')
            if i == 0:
                plt.title(f'Original (SSIM: {avg_ssim:.3f}, PSNR: {avg_psnr:.1f}dB)', fontsize=8)

            # Reconstructed image (bottom row)
            plt.subplot(rows, cols, cols + i + 1)
            recon_img = data[i].cpu()
            recon_img = torch.clamp(recon_img, 0, 1)
            recon_img = recon_img.permute(1, 2, 0).numpy()
            plt.imshow(recon_img)
            plt.axis('off')
            if i == 0:
                plt.title(f'Reconstructed (SSIM: {ssim_values[i]:.3f}, PSNR: {psnr_values[i]:.1f}dB)', fontsize=8)
    else:
        # Only show reconstructed images
        fig = plt.figure(figsize=(4, 4))
        for i in range(ind):
            plt.subplot(4, 4, i + 1)
            img = data[i].cpu()
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(sav_pth, name))
    plt.close()



def get_info_diff_loss(info_diff_type):
    if info_diff_type.lower() == 'l2':
        info_diff_loss = torch.nn.MSELoss(reduction='sum')
    elif info_diff_type.lower() == 'l1':
        info_diff_loss = torch.nn.SmoothL1Loss(reduction='sum', beta=1e-5)
    elif info_diff_type.lower() == 'sim':
        info_diff_loss = cos_sim
    else:
        ValueError(
            'info_diff_type: {} is not supported'.format(info_diff_type))
    return info_diff_loss


def get_reconstructor(atk_method, **kwargs):
    """

    Args:
        atk_method: the attack method name.
        **kwargs: other arguments

    Returns:

    """

    if atk_method.lower() == 'dlg':
        from federatedscope.attack.privacy_attacks.reconstruction_opt import\
            DLG
        logger.info(
            '--------- Getting reconstructor: DLG --------------------')

        return DLG(max_ite=kwargs['max_ite'],
                   lr=kwargs['lr'],
                   federate_loss_fn=kwargs['federate_loss_fn'],
                   device=kwargs['device'],
                   federate_lr=kwargs['federate_lr'],
                   optim=kwargs['optim'],
                   info_diff_type=kwargs['info_diff_type'],
                   federate_method=kwargs['federate_method'])
    elif atk_method.lower() == 'ig':
        from federatedscope.attack.privacy_attacks.reconstruction_opt import\
            InvertGradient
        logger.info(
            '------- Getting reconstructor: InvertGradient ------------------')
        return InvertGradient(max_ite=kwargs['max_ite'],
                              lr=kwargs['lr'],
                              federate_loss_fn=kwargs['federate_loss_fn'],
                              device=kwargs['device'],
                              federate_lr=kwargs['federate_lr'],
                              optim=kwargs['optim'],
                              info_diff_type=kwargs['info_diff_type'],
                              federate_method=kwargs['federate_method'],
                              alpha_TV=kwargs['alpha_TV'])
    elif atk_method.lower() == 'grnn':
        from federatedscope.attack.privacy_attacks.grnn_attack import \
            GRNNAttack
        logger.info('--------- Getting reconstructor: GRNN --------------------')
        return GRNNAttack(max_ite=kwargs['max_ite'],
                          lr=kwargs['lr'],
                          federate_loss_fn=kwargs['federate_loss_fn'],
                          device=kwargs['device'],
                          federate_method=kwargs['federate_method'],
                          federate_lr=kwargs['federate_lr'],
                          g_in=kwargs.get('g_in', 128),
                          tv_weight=kwargs.get('tv_weight', 1e-6),
                          use_wd=kwargs.get('use_wd', False),
                          dataset_name=kwargs.get('dataset_name', ''))
    else:
        raise ValueError(
            "attack method: {} lacks reconstructor implementation".format(
                atk_method))


def get_generator(dataset_name):
    '''
    Get the dataset's corresponding generator.
    Args:
        dataset_name: The dataset name; Type: str

    :returns:
        The generator; Type: object

    '''
    if dataset_name == 'femnist':
        from federatedscope.attack.models.gan_based_model import \
            GeneratorFemnist
        return GeneratorFemnist
    else:
        ValueError(
            "The generator to generate data like {} is not defined!".format(
                dataset_name))


def get_data_property(ctx):
    # A SHOWCASE for Femnist dataset: Property := whether contains a circle.
    x, label = [_.to(ctx.device) for _ in ctx.data_batch]

    prop = torch.zeros(label.size)
    positive_labels = [0, 6, 8]
    for ind in range(label.size()[0]):
        if label[ind] in positive_labels:
            prop[ind] = 1
    prop.to(ctx.device)
    return prop


def get_passive_PIA_auxiliary_dataset(dataset_name):
    '''

    Args:
        dataset_name (str): dataset name

    :returns:

    the auxiliary dataset for property inference attack. Type: dict

    {
        'x': array,
        'y': array,
        'prop': array
                    }

    '''
    for func in register.auxiliary_data_loader_PIA_dict.values():
        criterion = func(dataset_name)
        if criterion is not None:
            return criterion
    if dataset_name == 'toy':

        def _generate_data(instance_num=1000, feature_num=5, save_data=False):
            """
            Generate data in Runner format
            Args:
                instance_num:
                feature_num:
                save_data:

            Returns:
                {
                            'x': ...,
                            'y': ...,
                            'prop': ...
                        }

            """
            weights = np.random.normal(loc=0.0, scale=1.0, size=feature_num)
            bias = np.random.normal(loc=0.0, scale=1.0)

            prop_weights = np.random.normal(loc=0.0,
                                            scale=1.0,
                                            size=feature_num)
            prop_bias = np.random.normal(loc=0.0, scale=1.0)

            x = np.random.normal(loc=0.0,
                                 scale=0.5,
                                 size=(instance_num, feature_num))
            y = np.sum(x * weights, axis=-1) + bias
            y = np.expand_dims(y, -1)
            prop = np.sum(x * prop_weights, axis=-1) + prop_bias
            prop = 1.0 * ((1 / (1 + np.exp(-1 * prop))) > 0.5)
            prop = np.expand_dims(prop, -1)

            data_train = {'x': x, 'y': y, 'prop': prop}
            return data_train

        return _generate_data()
    else:
        ValueError(
            'The data cannot be loaded. Please specify the data load function.'
        )


def plot_mia_loss_compare(loss_in_pth, loss_out_pth, in_round=20):
    loss_in = np.loadtxt(loss_in_pth, delimiter=',')
    loss_out = np.loadtxt(loss_out_pth, delimiter=',')

    import matplotlib.pyplot as plt

    loss_in_all = []
    loss_out_all = []
    for i in range(len(loss_in)):
        if i == in_round:
            pass
        else:
            loss_in_all.append(loss_in[i])
            loss_out_all.append(loss_out[i])

    plt.plot(loss_out_all, label='not-in', alpha=0.9, color='red', linewidth=2)
    plt.plot(loss_in_all,
             linestyle=':',
             label='in',
             alpha=0.9,
             linewidth=2,
             color='blue')

    plt.legend()
    plt.xlabel('Round', fontsize=16)
    plt.ylabel('$L_x$', fontsize=16)
    plt.show()


def sav_officehome_image(data, sav_pth, name, original_data=None):
    """Save OfficeHome reconstructed images with optional original comparison

    Args:
        data: Reconstructed images [B, C, H, W]
        sav_pth: Save path
        name: Filename
        original_data: Optional original images for comparison [B, C, H, W]

    Note: GRNN generator outputs images in [0,1] range (sigmoid activation),
    so we don't need to denormalize. We display them directly.
    """
    if len(data.shape) == 3:  # single image [C, H, W]
        data = torch.unsqueeze(data, 0)

    ind = min(data.shape[0], 16)

    # Determine grid size based on whether we have original data
    if original_data is not None:
        # Show original and reconstructed side by side
        if len(original_data.shape) == 3:
            original_data = torch.unsqueeze(original_data, 0)

        # 计算SSIM和PSNR值
        ssim_values = []
        psnr_values = []
        for i in range(min(ind, 8)):
            # 原始图像已在 [0,1] 范围（ToTensor，无归一化），直接使用
            orig_img = original_data[i].cpu()
            orig_img = torch.clamp(orig_img, 0, 1)
            orig_img = orig_img.permute(1, 2, 0).numpy()

            # 准备重建图像
            recon_img = data[i].cpu()
            recon_img = torch.clamp(recon_img, 0, 1)
            recon_img = recon_img.permute(1, 2, 0).numpy()

            # 计算SSIM
            ssim_val = calculate_ssim(orig_img, recon_img, data_range=1.0, multichannel=True)
            ssim_values.append(ssim_val)

            # 计算PSNR
            psnr_val = calculate_psnr(orig_img, recon_img, data_range=1.0)
            psnr_values.append(psnr_val)

        # 计算平均SSIM和PSNR
        avg_ssim = np.mean(ssim_values) if ssim_values else -1.0
        avg_psnr = np.mean(psnr_values) if psnr_values else -1.0
        logger.info(f"[METRICS] OfficeHome reconstruction quality - SSIM: {avg_ssim:.4f}, PSNR: {avg_psnr:.2f} dB | Individual SSIM: {[f'{s:.3f}' for s in ssim_values]}, PSNR: {[f'{p:.2f}' for p in psnr_values]}")

        fig = plt.figure(figsize=(8, 4))
        rows, cols = 2, min(ind, 8)  # 2 rows: original + reconstructed

        for i in range(min(ind, 8)):
            # Original image (top row)
            plt.subplot(rows, cols, i + 1)
            orig_img = original_data[i].cpu()
            # 原始图像已在 [0,1] 范围（ToTensor，无归一化），直接使用
            orig_img = torch.clamp(orig_img, 0, 1)
            orig_img = orig_img.permute(1, 2, 0).numpy()
            plt.imshow(orig_img)
            plt.axis('off')
            if i == 0:
                plt.title(f'Original (SSIM: {avg_ssim:.3f}, PSNR: {avg_psnr:.1f}dB)', fontsize=8)

            # Reconstructed image (bottom row)
            plt.subplot(rows, cols, cols + i + 1)
            recon_img = data[i].cpu()
            recon_img = torch.clamp(recon_img, 0, 1)
            recon_img = recon_img.permute(1, 2, 0).numpy()
            plt.imshow(recon_img)
            plt.axis('off')
            if i == 0:
                plt.title(f'Reconstructed (SSIM: {ssim_values[i]:.3f}, PSNR: {psnr_values[i]:.1f}dB)', fontsize=8)
    else:
        # Only show reconstructed images
        fig = plt.figure(figsize=(4, 4))
        for i in range(ind):
            plt.subplot(4, 4, i + 1)
            img = data[i].cpu()
            img = torch.clamp(img, 0, 1)
            img = img.permute(1, 2, 0).numpy()
            plt.imshow(img)
            plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(sav_pth, name))
    plt.close()
