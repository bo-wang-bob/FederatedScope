import unittest
from federatedscope.cv.dataset.domainnet import DomainNet


class UploadedSplitTests(unittest.TestCase):
    def test_explicit_splits_do_not_change_with_random_seed(self):
        rows = [dict(path='train/A/a.jpg', label=0, split='train'),
                dict(path='train/B/b.jpg', label=1, split='train'),
                dict(path='val/A/c.jpg', label=0, split='val'),
                dict(path='test/B/d.jpg', label=1, split='test')]
        for seed in (1, 42, 12345):
            for split, count in [('train', 2), ('val', 1), ('test', 1)]:
                loader = object.__new__(DomainNet)
                loader.records, loader.root, loader.split, loader.seed = rows, '/fixture', split, seed
                images, labels = loader._load_data()
                self.assertEqual(len(images), count)
                self.assertEqual(labels, [r['label'] for r in rows if r['split'] == split])
                self.assertTrue(all(split + '/' in p.replace('\\', '/') for p in images))

    def test_mixed_split_contract_is_rejected(self):
        loader = object.__new__(DomainNet)
        loader.records = [dict(path='a.png', label=0, split='train'), dict(path='b.png', label=1)]
        with self.assertRaises(ValueError):
            loader._load_data()


if __name__ == '__main__':
    unittest.main()
