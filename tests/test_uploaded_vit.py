import json
import pytest
from federatedscope.standalone_api.uploaded_features import profile, extractor
from federatedscope.standalone_api.uploaded_prediction import feature_source


def checkpoint(kind, source=None):
    result = {'backbone': {'feature_extractor': kind}, 'architecture': {'input_dim': profile(kind)[1]}}
    if source is not None:
        result['uploadFeatureSource'] = source
    return result


@pytest.mark.parametrize('kind', ['cnn', 'clip'])
def test_checkpoint_selects_its_own_backbone(tmp_path, kind):
    source = dict(dataset='dataset-hash', weights='weight-hash', preprocessing=profile(kind)[2])
    selected, recorded = feature_source(checkpoint(kind, source), {'fingerprint': 'dataset-hash'}, tmp_path)
    assert selected == kind and recorded == source
    with pytest.raises(ValueError, match='不一致'):
        feature_source(checkpoint(kind, source), {'fingerprint': 'other-dataset'}, tmp_path)


def test_legacy_cnn_does_not_read_new_vit_cache(tmp_path):
    root = tmp_path / 'features'
    (root / 'vit-b16').mkdir(parents=True)
    source = dict(dataset='dataset-hash', weights='cnn-weight', preprocessing=profile('cnn')[2])
    (root / 'cnn.json').write_text(json.dumps({'source': source}))
    (root / 'vit-b16/clip.json').write_text(json.dumps({'source': {**source, 'weights': 'vit-weight', 'preprocessing': profile('clip')[2]}}))
    assert feature_source(checkpoint('cnn'), {'fingerprint': 'dataset-hash'}, tmp_path)[1]['weights'] == 'cnn-weight'
    with pytest.raises(ValueError, match='缺少训练时'):
        feature_source(checkpoint('clip'), {'fingerprint': 'dataset-hash'}, tmp_path)
    (root / 'another.json').write_text(json.dumps({'source': {**source, 'weights': 'different-cnn-weight'}}))
    with pytest.raises(ValueError, match='唯一确认'):
        feature_source(checkpoint('cnn'), {'fingerprint': 'dataset-hash'}, tmp_path)


def test_wrong_dimension_or_missing_weights_fail(tmp_path):
    source = dict(dataset='hash', weights='weight', preprocessing=profile('clip')[2])
    saved = checkpoint('clip', source)
    saved['architecture']['input_dim'] = 1024
    with pytest.raises(ValueError, match='维度不一致'):
        feature_source(saved, {'fingerprint': 'hash'}, tmp_path)
    with pytest.raises(ValueError, match='缺少本地预训练权重'):
        extractor(tmp_path, kind='clip')
