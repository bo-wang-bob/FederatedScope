"""Model paths remain portable without rewriting existing experiment artifacts."""
import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import yaml

from federatedscope.standalone_api.platform_config import ConfigFactory
from federatedscope.standalone_api.platform_paths import portable_config, resolve_path
from federatedscope.standalone_api.platform_privacy import PrivacyConfig
from federatedscope.standalone_api.privacy_artifacts import prepare_run

REPO = Path(__file__).resolve().parents[1]


class ModelRelativePathsTests(unittest.TestCase):
    def test_model_and_checkpoint_fields_are_relative_without_mutating_input(self):
        raw = {'ggeur': {key: str(REPO / 'resources/weights/model.pt') for key in (
            'clip_model_path', 'cnn_checkpoint_path', 'timm_checkpoint_path',
            'bert_model_path', 'bert_tokenizer_path', 'mlp_checkpoint_dir')},
            'federate': {'save_to': str(REPO / 'exp/final.pt'), 'restore_from': 'resources/initial.pt'}}
        before = copy.deepcopy(raw)
        converted = portable_config(raw, REPO)
        self.assertEqual(raw, before)
        for key, value in converted['ggeur'].items():
            self.assertFalse(Path(value).is_absolute())
            self.assertEqual(resolve_path(REPO, value), resolve_path(REPO, raw['ggeur'][key]))
        self.assertEqual(converted['federate'], {'save_to': 'exp/final.pt', 'restore_from': 'resources/initial.pt'})
        self.assertEqual(portable_config(converted, REPO), converted)

    def test_empty_optional_paths_stay_empty(self):
        raw = {'ggeur': {'cnn_checkpoint_path': '', 'clip_model_path': None}, 'federate': {'restore_from': ''}}
        self.assertEqual(portable_config(raw, REPO), raw)

    def test_relative_output_uses_backend_not_working_directory(self):
        previous = Path.cwd()
        with tempfile.TemporaryDirectory() as elsewhere, patch.dict(os.environ, {}, clear=True):
            try:
                os.chdir(elsewhere)
                base = ConfigFactory(REPO)
                request = base.normalize({'group': 'military_vit', 'method': 'fedavg'})
                config, _ = base.build(request, 'exp/path-test')
                self.assertEqual(config['ggeur']['mlp_checkpoint_dir'], 'exp/path-test/checkpoints')
                self.assertEqual(config['ggeur']['clip_model_path'], 'resources/models/ViT-B-16.pt')
                privacy = PrivacyConfig(base)
                for defense in (False, True):
                    config, provenance = privacy.build(privacy.defaults('military_cnn', defense), 'exp/privacy-test')
                    self.assertEqual(config['ggeur']['mlp_checkpoint_dir'], 'exp/privacy-test/checkpoints')
                    self.assertEqual(provenance['pathBase'], 'backend-directory')
            finally:
                os.chdir(previous)

    def test_export_rebases_models_as_well_as_dataset(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            repo = Path(directory) / 'backend'
            output = repo / 'exp/jobs' / ('a' * 32)
            output.mkdir(parents=True)
            raw = {'data': {'root': 'resources/datasets/images'},
                   'ggeur': {'cnn_checkpoint_path': 'resources/torch/hub/checkpoints/model.pth',
                             'mlp_checkpoint_dir': str(output / 'checkpoints')},
                   'federate': {'restore_from': 'resources/initial.pt'}, 'outdir': str(output / 'logs')}
            source = output / 'effective.yaml'
            source.write_text(yaml.safe_dump(raw), encoding='utf-8')
            original = source.read_bytes()
            archive = prepare_run(repo, {'group': 'military_cnn', 'defense': False}, 'a' * 32, output)
            exported = yaml.safe_load((archive / 'config.yaml').read_text(encoding='utf-8'))
            for section, keys in [('data', ['root']), ('ggeur', ['cnn_checkpoint_path', 'mlp_checkpoint_dir']),
                                  ('federate', ['restore_from'])]:
                for key in keys:
                    self.assertFalse(Path(exported[section][key]).is_absolute())
                    self.assertEqual(resolve_path(archive, exported[section][key]), resolve_path(repo, raw[section][key]))
            self.assertEqual(source.read_bytes(), original)
            self.assertEqual(exported['outdir'], '.')

    def test_privacy_templates_have_no_server_dataset_root(self):
        for file in (REPO / 'scripts/privacy_presets').glob('*.yaml'):
            raw = yaml.safe_load(file.read_text(encoding='utf-8'))
            self.assertFalse(Path(raw['data']['root']).is_absolute(), file.name)


if __name__ == '__main__':
    unittest.main()
