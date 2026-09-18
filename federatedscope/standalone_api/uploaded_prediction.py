"""Forward inference on labelled/unlabelled folders and single uploaded images."""
from pathlib import Path


def run(spec):
    import torch
    from scripts.test_outline_validation.evaluate_saved_mlp import build_model
    from .uploaded_datasets import DatasetStore
    from .uploaded_features import extractor, tensors
    from .platform_worker import emit, save, digest, classification_metrics
    import numpy as np
    repo = Path(__file__).resolve().parents[2]
    store = DatasetStore(repo)
    value = store.get(spec['request']['uploadId'])
    training = store.get(spec['trainingDatasetId'])
    seen = {r['sha256'] for r in training['items']}
    if any(row['sha256'] in seen for row in value['items']):
        raise ValueError('测试集含训练数据源中的图片，请使用独立测试图片；原有留出测试可在模型验证中选择')
    if digest(spec['checkpointPath']) != spec['checkpointHash']:
        raise ValueError('模型已改变，请重新选择模型')
    checkpoint = torch.load(spec['checkpointPath'], map_location='cpu', weights_only=True)
    model = build_model(checkpoint['architecture'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    torch.set_num_threads(2)
    backbone, weight = extractor(repo)
    # Upload training always uses this local frozen backbone, checked by preflight.
    cache_metadata = list((store.directory(training['id']) / 'features').glob('*.json'))
    if not cache_metadata:
        raise ValueError('缺少训练时的骨干权重指纹')
    import json
    expected = json.loads(cache_metadata[0].read_text(encoding='utf-8'))['source']['weights']
    if digest(weight) != expected:
        raise ValueError('骨干权重与训练时不同，拒绝使用不同特征空间')
    names, rows = spec['classNames'], []
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    with torch.inference_mode():
        for start in range(0, value['count'], 8):
            indices = list(range(start, min(start + 8, value['count'])))
            files = [store.image(value['id'], i) for i in indices]
            for i, file in zip(indices, files):
                if digest(file) != value['items'][i]['sha256']:
                    raise ValueError('测试图片内容已改变')
            logits = model(backbone(tensors(files)).float())
            if logits.shape[1] != len(names) or not torch.isfinite(logits).all():
                raise ValueError('模型输出类别或数值非法')
            scores = torch.softmax(logits, dim=-1)
            for i, score in zip(indices, scores):
                predicted = int(score.argmax())
                item = value['items'][i]
                label = item['path'].split('/')[0] if value['classes'] else None
                if label is not None:
                    matrix[names.index(label), predicted] += 1
                rows.append(dict(index=i, filename=item.get('originalPath', item['path']), labelName=label, predictedName=names[predicted],
                    confidence=float(score[predicted]), correct=None if label is None else label == names[predicted],
                    imageUrl=f'/api/platform/datasets/{value["id"]}/images/{i}'))
            emit('stage', stage=f'测试图片推理 {len(rows)}/{value["count"]}')
    save(Path(spec['output']) / 'result.json', dict(items=rows, samples=len(rows),
        metrics=classification_metrics(matrix) if value['classes'] else None,
        labelled=bool(value['classes']), checkpointSha256=spec['checkpointHash']))
