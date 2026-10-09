import json
import torch
import pytest

from federatedscope.standalone_api.platform_inference import classifier_for_domain, snapshot_domain_heads
from federatedscope.standalone_api.platform_worker import evaluate, predict, digest


def fixture(tmp_path):
    model = torch.nn.Linear(2, 2)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))
        model.bias.zero_()
    state = {'Art': {'weight': torch.tensor([[0., 1.], [1., 0.]]), 'bias': torch.zeros(2)}}
    extension = snapshot_domain_heads(state)
    architecture = {'model_type': 'ggeur_mlp', 'input_dim': 2, 'hidden_dim': 0, 'num_classes': 2}
    checkpoint = dict(architecture=architecture, state_dict=model.state_dict(),
                      inferenceHeads=extension, dataset='fixture', backbone={'test': 'frozen'})
    bundle = dict(dataset='fixture', backbone={'test': 'frozen'},
                  features={'Art': torch.tensor([[3., 0.], [0., 4.]])}, labels={'Art': torch.tensor([1, 0])})
    torch.save(checkpoint, tmp_path / 'head.pt')
    torch.save(bundle, tmp_path / 'test.pt')
    spec = {'checkpointPath': str(tmp_path / 'head.pt'), 'checkpointHash': digest(tmp_path / 'head.pt'),
            'bundlePath': str(tmp_path / 'test.pt'), 'bundleHash': digest(tmp_path / 'test.pt'),
            'output': str(tmp_path), 'request': {}}
    return model, state, extension, spec


def test_original_head_and_snapshot_are_preserved(tmp_path):
    model, state, extension, _ = fixture(tmp_path)
    assert classifier_for_domain(model, 'Art', None, 2, 2) is model
    assert classifier_for_domain(model, 'Other', extension, 2, 2) is model
    state['Art']['weight'].zero_()
    assert extension['domains']['Art']['weight'].sum() == 2
    head = classifier_for_domain(model, 'Art', extension, 2, 2)
    torch.testing.assert_close(head(torch.tensor([[30., 0.]])), torch.tensor([[0., 1.]]))


def test_evaluation_and_single_prediction_reload_same_domain_head(tmp_path):
    _, _, _, spec = fixture(tmp_path)
    evaluate(spec)
    metrics = json.loads((tmp_path / 'result.json').read_text())
    assert metrics['accuracy'] == 1.0
    manifest = tmp_path / 'data_manifest.json'
    manifest.write_text(json.dumps({'test': {'Art': {'a.jpg': 1, 'b.jpg': 0}}}))
    image = tmp_path / 'a.jpg'
    image.write_bytes(b'fixture original image')
    spec.update(manifestPath=str(manifest), manifestHash=digest(manifest),
                classNames=['A', 'B'], testProvenance='fixture',
                sample={'domain': 'Art', 'index': 0, 'key': 'a.jpg', 'label': 1,
                        'imagePath': str(image), 'id': 'sample-a', 'filename': 'a.jpg'})
    spec['request']['imageSha256'] = digest(image)
    predict(spec)
    prediction = json.loads((tmp_path / 'result.json').read_text())
    assert prediction['predictedClass'] == 1 and prediction['correct']
    assert prediction['confidence'] == pytest.approx(torch.softmax(torch.tensor([0., 1.]), 0)[1].item())


def test_invalid_domain_weights_fail_closed(tmp_path):
    model, _, extension, _ = fixture(tmp_path)
    extension['domains']['Art']['weight'][0, 0] = float('nan')
    with pytest.raises(ValueError):
        classifier_for_domain(model, 'Art', extension, 2, 2)
    with pytest.raises(ValueError):
        classifier_for_domain(model, 'Art', {'version': 999}, 2, 2)
