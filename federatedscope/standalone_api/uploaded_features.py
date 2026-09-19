"""Local ViT/CNN extraction with separate feature identities for uploaded images."""
import json
from pathlib import Path

from .uploaded_datasets import DatasetStore


def profile(kind):
    if kind == 'clip':
        return 'resources/models/ViT-B-16.pt', 512, 'openclip-vit-b16-openai-eval-float32-v1'
    if kind == 'cnn':
        return 'resources/torch/hub/checkpoints/convnext_base-6075fbad.pth', 1024, 'resize224-imagenet-float32-v1'
    raise ValueError('不支持的上传模型骨干：' + str(kind))


def extractor(repo, device='cpu', kind='cnn'):
    import torch
    relative, _, _ = profile(kind)
    weight = Path(repo) / relative
    if not weight.is_file():
        raise ValueError(f'缺少本地预训练权重：{weight}')
    if kind == 'clip':
        import open_clip
        clip, _, operation = open_clip.create_model_and_transforms('ViT-B-16', pretrained=str(weight), device=device)
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = clip
                self.upload_transform = operation
            def forward(self, images):
                return self.model.encode_image(images, normalize=False)
        model = Encoder().to(device).eval()
        model.requires_grad_(False)
        return model, weight
    from federatedscope.contrib.model.ggeur_cnn_extractor import CNNFeatureExtractor
    model = CNNFeatureExtractor('convnext_base', pretrained=False, freeze=True, checkpoint_path=str(weight)).to(device).eval()
    return model, weight


def transform():
    from torchvision import transforms
    return transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225])])


def tensors(files, operation=None):
    import torch
    from PIL import Image
    operation = operation or transform()
    values = []
    for file in files:
        with Image.open(file) as image:
            values.append(operation(image.convert('RGB')))
    return torch.stack(values)


def prepare_uploaded(cfg, group):
    import numpy as np
    import torch
    from federatedscope.contrib.worker.ggeur_client import GGEURClient
    from .platform_worker import emit, digest, save
    repo = Path(__file__).resolve().parents[2]
    store = DatasetStore(repo)
    value = store.training(group)
    # Older saved cloud-based specs disabled caching. Uploaded data always has
    # an isolated verified cache, including when replaying those older specs.
    cfg.ggeur.use_feature_cache = True
    cfg.ggeur.require_complete_feature_cache = True
    kind = cfg.ggeur.feature_extractor
    relative, dimension, preprocessing = profile(kind)
    weight = repo / relative
    if not weight.is_file():
        raise ValueError('缺少本地预训练权重：' + relative)
    probe = object.__new__(GGEURClient)
    probe._cfg, probe.ggeur_cfg = cfg, cfg.ggeur
    probe.embedding_dim, probe.feature_extractor_type = dimension, kind
    cache_path = probe._get_feature_cache_path('uploaded')
    if not cache_path:
        raise ValueError('上传数据集的独立特征缓存路径未配置，请重新预检')
    target = Path(cache_path)
    metadata = target.with_suffix('.json')
    fingerprint = dict(dataset=value['fingerprint'], weights=digest(weight), preprocessing=preprocessing)
    files = [store.image(value['id'], i) for i in range(value['count'])]
    for file, row in zip(files, value['items']):
        if digest(file) != row['sha256']:
            raise ValueError('上传图片被修改，请重新上传为新版本')
    if target.is_file() and metadata.is_file():
        saved = json.loads(metadata.read_text(encoding='utf-8'))
        if saved.get('source') == fingerprint and saved.get('sha256') == digest(target):
            emit('stage', stage='复用此上传数据集已核验的本地特征缓存')
            return fingerprint
    device = f'cuda:{cfg.device}' if cfg.use_gpu else 'cpu'
    if cfg.use_gpu and not torch.cuda.is_available():
        raise ValueError('CUDA 不可用；普通训练可选择 CPU，隐私训练需要预设 GPU')
    model, _ = extractor(repo, device, kind)
    blocks = []
    with torch.inference_mode():
        for start in range(0, len(files), 8):
            block = model(tensors(files[start:start + 8], getattr(model, 'upload_transform', None)).to(device)).float().cpu().numpy()
            if not np.isfinite(block).all():
                raise ValueError('图片特征包含无效数值')
            blocks.append(block)
            emit('stage', stage=f'提取上传数据集特征 {min(start + 8, len(files))}/{len(files)}')
    target.parent.mkdir(parents=True, exist_ok=True)
    # Completed file is published only after all samples were extracted.
    import uuid
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, paths=np.asarray([probe._feature_cache_key(str(f)) for f in files]), features=np.concatenate(blocks))
    temporary.replace(target)
    save(metadata, dict(source=fingerprint, sha256=digest(target)))
    return fingerprint
