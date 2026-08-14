# -*-coding:utf-8-*-
"""
GRNN (Gradient Reconstruction Neural Network) Attack

基于生成器的梯度反演攻击，使用神经网络直接从梯度重建训练数据

References:
    使用生成对抗网络进行梯度反演，重建客户端私有训练数据
"""

import torch
import torch.nn as nn
import numpy as np
import logging

logger = logging.getLogger(__name__)


class GLU(nn.Module):
    """Gated Linear Unit"""
    def __init__(self):
        super(GLU, self).__init__()

    def forward(self, x):
        nc = x.size(1)
        assert nc % 2 == 0, 'channels dont divide 2!'
        nc = int(nc / 2)
        return x[:, :nc] * torch.sigmoid(x[:, nc:])


class GRNNGenerator(nn.Module):
    """
    GRNN 生成器网络

    输入：随机噪声 z (shape: [batch_size, g_in])
    输出：
        - 重建图像 (shape: [batch_size, channel, H, W])
        - 重建标签 (shape: [batch_size, num_classes])
    """
    def __init__(self, num_classes, shape_img, batchsize, channel=3, g_in=128, d=32):
        super(GRNNGenerator, self).__init__()
        self.g_in = g_in
        self.batchsize = batchsize
        self.target_shape = shape_img  # 保存目标图像尺寸

        # 标签生成分支
        self.fc2 = nn.Sequential(
            nn.Linear(g_in, num_classes)
        )

        # 图像生成分支
        # 使用最接近的 2 的幂来构建生成器
        nearest_power_of_2 = 2 ** int(np.log2(shape_img) + 0.5)
        block_num = int(np.log2(nearest_power_of_2) - 3)

        self.block0 = nn.Sequential(
            nn.ConvTranspose2d(g_in, d * pow(2, block_num) * 2, 4, 1, 0),
            GLU()
        )

        self.blocks = nn.ModuleList()
        for bn in reversed(range(block_num)):
            self.blocks.append(self.upBlock(pow(2, bn + 1) * d, pow(2, bn) * d))
        self.deconv_out = self.upBlock(d, channel)

        # 计算生成器的实际输出尺寸
        self.generated_size = 4 * (2 ** (block_num + 1))

    @staticmethod
    def upBlock(in_planes, out_planes):
        def conv3x3(in_planes, out_planes):
            return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=1,
                           padding=1, bias=False)

        block = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            conv3x3(in_planes, out_planes * 2),
            nn.BatchNorm2d(out_planes * 2),
            GLU()
        )
        return block

    def weight_init(self, mean=0.0, std=0.02):
        """初始化权重"""
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    def forward(self, x):
        """
        Args:
            x: 输入噪声 [batch_size, g_in]

        Returns:
            output: 生成图像 [batch_size, channel, H, W]
            y: 生成标签 (softmax) [batch_size, num_classes]
        """
        y = torch.softmax(self.fc2(x), 1)
        x = x.view(self.batchsize, self.g_in, 1, 1)
        output = self.block0(x)

        for block in self.blocks:
            output = block(output)
        output = self.deconv_out(output)
        output = torch.sigmoid(output)

        # 如果生成器输出尺寸与目标尺寸不匹配，使用插值调整
        if self.generated_size != self.target_shape:
            output = torch.nn.functional.interpolate(
                output,
                size=(self.target_shape, self.target_shape),
                mode='bilinear',
                align_corners=False
            )

        return output, y


def normal_init(m, mean, std):
    """权重初始化辅助函数"""
    if isinstance(m, (nn.ConvTranspose2d, nn.Conv2d, nn.Linear)):
        m.weight.data.normal_(mean, std)
        if m.bias is not None:
            m.bias.data.zero_()


class TVLoss(nn.Module):
    """Total Variation Loss - 用于图像平滑正则化"""
    def __init__(self, weight=1.0):
        super(TVLoss, self).__init__()
        self.weight = weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])

        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x-1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x-1]), 2).sum()

        return self.weight * 2 * (h_tv/count_h + w_tv/count_w) / batch_size

    @staticmethod
    def _tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


class GRNNAttack(object):
    """
    GRNN 梯度反演攻击主类

    从接收到的梯度信息重建客户端的训练数据

    Args:
        max_ite (int): 最大迭代次数
        lr (float): 生成器学习率
        federate_loss_fn (callable): 联邦学习使用的损失函数
        device (str): 运行设备
        federate_method (str): 联邦学习方法 (e.g., 'fedavg')
        federate_lr (float): FL 训练学习率
        g_in (int): 生成器输入维度
        tv_weight (float): TV 正则化权重
        use_wd (bool): 是否使用 Wasserstein 距离
    """

    def __init__(self,
                 max_ite,
                 lr,
                 federate_loss_fn,
                 device,
                 federate_method,
                 federate_lr=None,
                 g_in=128,
                 tv_weight=1e-6,
                 use_wd=False,
                 dataset_name=''):

        if federate_method.lower() == "fedavg":
            assert federate_lr is not None

        self.info_is_para = federate_method.lower() == "fedavg"
        self.federate_lr = federate_lr

        self.max_ite = max_ite
        self.lr = lr
        self.device = device
        self.federate_loss_fn = federate_loss_fn
        self.g_in = g_in
        self.tv_weight = tv_weight
        self.use_wd = use_wd
        self.dataset_name = dataset_name.lower() if dataset_name else ''
        self.generator = None
        self.g_optimizer = None
        self.tv_loss = TVLoss()

        self.loss_trace = []

        # 兼容 PassiveServer（DLG 接口）
        self.dlg_recover_loss = 0.0

    def eval(self):
        pass

    def _get_normalization_stats(self):
        if 'cifar10' in self.dataset_name:
            mean = torch.tensor([0.4914, 0.4822, 0.4465],
                                device=self.device).view(1, 3, 1, 1)
            std = torch.tensor([0.2470, 0.2435, 0.2616],
                               device=self.device).view(1, 3, 1, 1)
            return mean, std

        if 'mnist' in self.dataset_name or 'femnist' in self.dataset_name or 'fashionmnist' in self.dataset_name:
            mean = torch.tensor([0.9637], device=self.device).view(1, 1, 1, 1)
            std = torch.tensor([0.1592], device=self.device).view(1, 1, 1, 1)
            return mean, std

        return None, None

    def _prepare_model_input(self, fake_data):
        mean, std = self._get_normalization_stats()
        if mean is None or std is None:
            return fake_data
        return (fake_data - mean) / std

    def _flatten_gradients(self, gradients):
        """将梯度列表展平为一维向量"""
        flatten_grad = None
        for layer_g in gradients:
            if flatten_grad is None:
                flatten_grad = torch.flatten(layer_g)
            else:
                flatten_grad = torch.cat((flatten_grad, torch.flatten(layer_g)))
        return flatten_grad

    def get_original_gradient_from_para(self, model, original_info, model_para_name):
        """
        从参数更新转换为梯度

        基于公式: P_t = P - η*g
        """
        original_gradient = [
            ((original_para - original_info[name].to(torch.device(self.device))) /
             self.federate_lr).detach()
            for original_para, name in zip(model.parameters(), model_para_name)
        ]
        return original_gradient

    def reconstruct(self, model, original_info, data_feature_dim, num_class, batch_size, real_gradients=None):
        """
        执行 GRNN 重建攻击

        Args:
            model: FL 模型
            original_info: 接收到的梯度/参数更新
            data_feature_dim: 数据特征维度 (e.g., [3, 32, 32])
            num_class: 类别数量
            batch_size: 批次大小
            real_gradients: 真实梯度（可选），如果提供则优先使用，否则从参数反推

        Returns:
            reconstructed_data: 重建的数据 [batch_size, *data_feature_dim]
            reconstructed_label: 重建的标签 [batch_size]
        """

        logger.info("="*60)
        logger.info("Starting GRNN Gradient Inversion Attack")
        logger.info(f"Max iterations: {self.max_ite}, LR: {self.lr}, G_in: {self.g_in}")
        if real_gradients is not None:
            logger.info("Using REAL GRADIENTS from client")
        else:
            logger.info("Computing gradients from parameter difference")
        logger.info("="*60)

        # 初始化生成器
        if len(data_feature_dim) == 3:
            shape_img = data_feature_dim[1]  # 假设是正方形图像
            channel = data_feature_dim[0]
        else:
            raise ValueError(f"Unsupported data dimension: {data_feature_dim}")

        self.generator = GRNNGenerator(
            num_classes=num_class,
            shape_img=shape_img,
            batchsize=batch_size,
            channel=channel,
            g_in=self.g_in
        ).to(self.device)

        self.generator.weight_init(mean=0.0, std=0.02)

        # 优化器
        self.g_optimizer = torch.optim.RMSprop(
            self.generator.parameters(),
            lr=self.lr,
            momentum=0.99
        )

        # 获取梯度参数名称
        para_trainable_name = [p[0] for p in model.named_parameters()]

        # 优先使用真实梯度，如果没有则从参数反推
        if real_gradients is not None:
            # 使用客户端发送的真实梯度
            logger.info(f"Using real gradients with {len(real_gradients)} parameters")
            original_gradient = []
            missing_grad_num = 0
            for name, param in model.named_parameters():
                if name in real_gradients:
                    original_gradient.append(
                        real_gradients[name].to(torch.device(self.device)))
                else:
                    original_gradient.append(torch.zeros_like(param))
                    missing_grad_num += 1
            if missing_grad_num > 0:
                logger.warning(
                    f'GRNN: {missing_grad_num} real-gradient parameters are missing; '
                    f'filled them with zeros for shape alignment.')
        elif self.info_is_para:
            # 从参数更新反推梯度
            logger.info("Computing gradients from parameter difference (FedAvg mode)")
            original_gradient = self.get_original_gradient_from_para(
                model, original_info, model_para_name=para_trainable_name
            )
        else:
            # 使用传入的梯度信息
            logger.info("Using gradients from original_info")
            original_gradient = [
                grad.to(torch.device(self.device)) for k, grad in original_info
            ]

        # 展平真实梯度
        flatten_true_g = self._flatten_gradients(original_gradient)

        # 初始化生成器输入
        G_ran_in = torch.randn(batch_size, self.g_in).to(self.device)

        # 优化循环
        self.loss_trace = []
        best_loss = float('inf')
        best_fake_data = None
        best_fake_label = None

        for iteration in range(self.max_ite):
            # 生成假数据和标签
            fake_data, fake_label = self.generator(G_ran_in)

            # 按训练时的数据预处理对齐输入范围，避免梯度匹配目标失配
            fake_model_input = self._prepare_model_input(fake_data)
            pred = model(fake_model_input)

            # 交叉熵损失 (与 GRNN 原实现一致)
            fake_loss = -torch.mean(
                torch.sum(
                    fake_label * torch.log(torch.softmax(pred, 1) + 1e-10),
                    dim=-1
                )
            )

            # 计算假梯度
            model_params = [p for p in model.parameters()]
            raw_fake_gradient = torch.autograd.grad(
                fake_loss,
                model_params,
                create_graph=True,
                allow_unused=True
            )
            fake_gradient = []
            unused_param_num = 0
            for param, grad in zip(model_params, raw_fake_gradient):
                if grad is None:
                    fake_gradient.append(torch.zeros_like(param))
                    unused_param_num += 1
                else:
                    fake_gradient.append(grad)
            if unused_param_num > 0 and iteration == 0:
                logger.warning(
                    f'GRNN: {unused_param_num} model parameters are unused in the current graph; '
                    f'their gradients are filled with zeros for matching.')

            flatten_fake_g = self._flatten_gradients(fake_gradient)

            # L2 梯度距离
            grad_diff_l2 = ((flatten_fake_g - flatten_true_g) ** 2).sum()

            # Wasserstein 距离 (可选)
            grad_diff_wd = 0
            if self.use_wd:
                try:
                    from federatedscope.attack.auxiliary.utils import wasserstein_distance
                    grad_diff_wd = wasserstein_distance(
                        flatten_fake_g.view(1, -1),
                        flatten_true_g.view(1, -1),
                        device=self.device
                    )
                except Exception as e:
                    if iteration == 0:
                        logger.warning(f"Wasserstein distance calculation failed: {e}")
                    grad_diff_wd = 0

            # TV 正则化
            tvloss = self.tv_weight * self.tv_loss(fake_data)

            # 总损失
            grad_diff = grad_diff_l2 + grad_diff_wd + tvloss

            # 反向传播
            self.g_optimizer.zero_grad()
            grad_diff.backward()
            self.g_optimizer.step()

            # 记录损失
            loss_val = grad_diff.detach().cpu().item()
            self.loss_trace.append(loss_val)

            if loss_val < best_loss:
                best_loss = loss_val
                best_fake_data = fake_data.detach().clone()
                best_fake_label = fake_label.detach().clone()

            # 日志输出
            if (iteration + 1 == self.max_ite) or iteration % 100 == 0:
                logger.info(
                    f'Iteration: {iteration}/{self.max_ite}, '
                    f'Loss: {grad_diff.item():.6f}, '
                    f'L2: {grad_diff_l2.item():.6f}, '
                    f'WD: {grad_diff_wd if isinstance(grad_diff_wd, (int, float)) else grad_diff_wd.item():.6f}, '
                    f'TV: {tvloss.item():.6f}'
                )

        logger.info("="*60)
        logger.info("GRNN Attack Completed")
        logger.info("="*60)

        # 更新最终损失（用于 PassiveServer 日志）
        self.dlg_recover_loss = best_loss if self.loss_trace else 0.0

        # 返回重建结果：使用整个优化过程中 loss 最低的那一步
        reconstructed_data = best_fake_data if best_fake_data is not None else fake_data.detach()
        best_label = best_fake_label if best_fake_label is not None else fake_label.detach()
        reconstructed_label = best_label.argmax(dim=1).detach()

        return reconstructed_data, reconstructed_label
