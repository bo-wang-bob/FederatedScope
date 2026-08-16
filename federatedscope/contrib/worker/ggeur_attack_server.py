import copy
import logging
import os

import torch

from federatedscope.contrib.worker.ggeur_server import GGEURServer
from federatedscope.core.auxiliaries.criterion_builder import get_criterion
from federatedscope.core.monitoring.events import emit_training_event
from federatedscope.attack.auxiliary.utils import get_data_info, \
    get_data_sav_fn, get_reconstructor

logger = logging.getLogger(__name__)


class GGEURPassiveServer(GGEURServer):
    """GGEUR server with passive GRNN reconstruction support for image branches."""

    def __init__(self,
                 ID=-1,
                 state=0,
                 data=None,
                 model=None,
                 client_num=5,
                 total_round_num=10,
                 device='cpu',
                 strategy=None,
                 state_to_reconstruct=None,
                 client_to_reconstruct=None,
                 **kwargs):
        super(GGEURPassiveServer, self).__init__(
            ID=ID,
            state=state,
            data=data,
            model=model,
            client_num=client_num,
            total_round_num=total_round_num,
            device=device,
            strategy=strategy,
            **kwargs)

        self.atk_method = self._cfg.attack.attack_method
        self.state_to_reconstruct = state_to_reconstruct
        self.client_to_reconstruct = client_to_reconstruct
        self.reconstruct_data = {}
        self.full_msg_buffer = {}

        self.model_criterion = get_criterion(self._cfg.criterion.type,
                                             device=self.device)
        self.data_dim, self.num_class, self.is_one_hot_label = get_data_info(
            self._cfg.data.type)
        self.reconstructor = self._get_reconstructor()
        self.reconstructed_data_sav_fn = get_data_sav_fn(self._cfg.data.type)
        self.reconstruct_data_summary = {}

    def _get_reconstructor(self):
        reconstructor_kwargs = {
            'max_ite': self._cfg.attack.max_ite,
            'lr': self._cfg.attack.reconstruct_lr,
            'federate_loss_fn': self.model_criterion,
            'device': self.device,
            'federate_lr': getattr(self._cfg.ggeur, 'cnn_lr', self._cfg.train.optimizer.lr),
            'optim': self._cfg.attack.reconstruct_optim,
            'info_diff_type': self._cfg.attack.info_diff_type,
            'federate_method': self._cfg.federate.method,
            'alpha_TV': self._cfg.attack.alpha_TV,
        }
        if self.atk_method.lower() == 'grnn':
            reconstructor_kwargs.update({
                'g_in': self._cfg.attack.get('grnn_g_in', 128),
                'tv_weight': self._cfg.attack.get('grnn_tv_weight', 1e-6),
                'use_wd': self._cfg.attack.get('grnn_use_wd', False),
                'dataset_name': self._cfg.data.type,
            })
        return get_reconstructor(self.atk_method, **reconstructor_kwargs)

    def _select_attack_payload(self, model_para, gradients):
        if isinstance(model_para, dict):
            if 'cnn' in model_para and model_para.get('cnn') is not None:
                attack_model = copy.deepcopy(self.global_cnn)
                attack_para = model_para['cnn']
                attack_grad = gradients.get('cnn') if isinstance(gradients, dict) else None
                return attack_model, attack_para, attack_grad
            if 'cnn_backbone' in model_para and model_para.get('cnn_backbone') is not None:
                attack_model = copy.deepcopy(self.global_cnn)
                attack_para = model_para['cnn_backbone']
                attack_grad = gradients.get('cnn_backbone') if isinstance(gradients, dict) else None
                return attack_model, attack_para, attack_grad
        return None, None, None

    def _reconstruct(self, model_para, batch_size, gradients, state, sender,
                     last_batch_data=None):
        logger.info('-------- reconstruct round:{}, client:{}---------'.format(
            state, sender))

        attack_model, attack_para, attack_gradients = self._select_attack_payload(
            model_para, gradients)
        if attack_model is None or attack_para is None:
            logger.warning(
                'GRNN attack skipped for round=%s, client=%s because the '
                'update does not contain an image-branch payload.', state,
                sender)
            emit_training_event(
                'warning.raised', level='warning', round=int(state),
                clientIndex=int(sender),
                message='重建攻击缺少图像分支更新，已跳过该节点')
            return

        if attack_gradients is not None:
            logger.info('Using real image-branch gradients for GRNN attack')
        else:
            logger.info('No real image-branch gradients available, will infer from parameters')

        dummy_data, dummy_label = self.reconstructor.reconstruct(
            model=attack_model.to(torch.device(self.device)),
            original_info=attack_para,
            real_gradients=attack_gradients,
            data_feature_dim=self.data_dim,
            num_class=self.num_class,
            batch_size=batch_size)

        if state not in self.reconstruct_data:
            self.reconstruct_data[state] = {}
        self.reconstruct_data[state][sender] = [dummy_data.cpu(),
                                                dummy_label.cpu()]

        reconstruction_dir = os.path.join(
            str(self._cfg.outdir), "grnn_reconstructions")
        os.makedirs(reconstruction_dir, exist_ok=True)
        reconstruction_path = os.path.join(
            reconstruction_dir, f"state_{int(state)}_client_{int(sender)}.pt")
        torch.save({
            "images": dummy_data.detach().cpu(),
            "labels": dummy_label.detach().cpu(),
            "state": int(state),
            "client_id": int(sender),
            "reconstruction_loss": float(self.reconstructor.dlg_recover_loss),
        }, reconstruction_path)
        logger.info("[GRNN] Saved reconstruction tensor: %s",
                    reconstruction_path)

        logger.info('Finish %s attack; Final reconstruction loss: %s',
                    self.atk_method.upper(),
                    self.reconstructor.dlg_recover_loss)

        metric_payload = {
            'round': int(state),
            'reconstructionLoss': float(
                self.reconstructor.dlg_recover_loss),
        }
        original_data = None
        if isinstance(last_batch_data, tuple) and len(last_batch_data) == 2:
            original_data = last_batch_data[0]
        if original_data is not None:
            reference = original_data.detach().float().cpu()
            reconstructed = dummy_data.detach().float().cpu()
            if reference.ndim == 3:
                reference = reference.unsqueeze(0)
            if reconstructed.ndim == 3:
                reconstructed = reconstructed.unsqueeze(0)
            count = min(reference.shape[0], reconstructed.shape[0])
            if count:
                reference = reference[:count].clamp(0.0, 1.0)
                reconstructed = reconstructed[:count].clamp(0.0, 1.0)
                if reference.shape[-2:] != reconstructed.shape[-2:]:
                    reference = torch.nn.functional.interpolate(
                        reference, size=reconstructed.shape[-2:],
                        mode='bilinear', align_corners=False)
                mse = (reference - reconstructed).pow(2).flatten(1).mean(1)
                psnr = 10.0 * torch.log10(
                    1.0 / mse.clamp_min(1.0e-12))
                metric_payload['reconstructionPsnr'] = float(
                    psnr.mean().item())
        emit_training_event('metric.updated', **metric_payload)

        if self.reconstructed_data_sav_fn is not None:
            label_str = '_'.join(map(str, dummy_label.cpu().numpy().tolist()))
            filename = f'image_state_{state}_client_{sender}_label_{label_str}.png'
            self.reconstructed_data_sav_fn(
                data=dummy_data.cpu(),
                sav_pth=self._cfg.outdir,
                name=filename,
                original_data=original_data)

    def run_reconstruct(self, state_list=None, sender_list=None):
        if state_list is None:
            state_list = self.full_msg_buffer.keys()

        for state in state_list:
            sender_iter = sender_list
            if sender_iter is None:
                sender_iter = self.full_msg_buffer.get(state, {}).keys()
            for sender in sender_iter:
                content = self.full_msg_buffer.get(state, {}).get(sender)
                if content is None:
                    continue
                sample_size = content[0]
                model_para = content[1] if len(content) >= 2 else None
                gradients = content[2] if len(content) >= 3 else None
                last_batch_data = content[3] if len(content) >= 4 else None
                recon_batch_size = self._cfg.dataloader.batch_size
                self._reconstruct(model_para=model_para,
                                  batch_size=recon_batch_size,
                                  gradients=gradients,
                                  state=state,
                                  sender=sender,
                                  last_batch_data=last_batch_data)

    def callback_funcs_model_para(self, message):
        if self.is_finish:
            return 'finish'

        round_idx, sender, content = message.state, message.sender, message.content
        self.sampler.change_state(sender, 'idle')
        if round_idx not in self.msg_buffer['train']:
            self.msg_buffer['train'][round_idx] = []
        if round_idx not in self.full_msg_buffer:
            self.full_msg_buffer[round_idx] = {}
        self.full_msg_buffer[round_idx][sender] = content

        if isinstance(content, tuple) and len(content) >= 2:
            sample_size, model_para = content[0], content[1]
        else:
            sample_size, model_para = 0, content

        self.msg_buffer['train'][round_idx].append((sample_size, model_para, sender))
        received_num = len(self.msg_buffer['train'][round_idx])
        emit_training_event(
            'client.status.changed', clientIndex=int(sender),
            status='上传完成', progress=100, round=int(round_idx),
            sampleCount=int(sample_size))
        if received_num == 1:
            emit_training_event(
                'stage.changed', round=int(round_idx), phaseIndex=3,
                stage='client_upload_projection')

        if self.state_to_reconstruct is None or round_idx in self.state_to_reconstruct:
            if self.client_to_reconstruct is None or sender in self.client_to_reconstruct:
                self.run_reconstruct(state_list=[round_idx], sender_list=[sender])

        expected_num = len(getattr(
            self, 'current_round_clients', [])) or self._client_num
        logger.info(
            f"Server: Received model from client {sender} for round {round_idx} "
            f"({received_num}/{expected_num})")

        if received_num >= expected_num:
            emit_training_event(
                'stage.changed', round=int(round_idx), phaseIndex=4,
                stage='domain_aggregation_projection')
            emit_training_event(
                'stage.changed', round=int(round_idx), phaseIndex=5,
                stage='domain_upload_projection')
            self._perform_fedavg(round_idx)
