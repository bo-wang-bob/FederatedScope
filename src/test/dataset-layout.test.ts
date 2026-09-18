import { expect, it } from 'vitest';
import { inspectDatasetFolder } from '../platform/datasetLayout';

const training = ['ants', 'bees'].flatMap(c => [1, 2, 3].map(i => `${c}/${i}.png`));
it('supports existing class-folder uploads', () => {
  expect(inspectDatasetFolder(training)).toMatchObject({ layout: 'classes', classes: ['ants', 'bees'], trainCount: 6 });
});
it.each(['test', 'val', 'valid', 'validation'])('recognizes train and %s from a whole dataset', folder => {
  expect(inspectDatasetFolder([...training.map(p => 'train/' + p), `${folder}/ants/a.png`]))
    .toMatchObject({ layout: 'split', classes: ['ants', 'bees'], trainCount: 6, testCount: 1, testSource: folder });
});
it('keeps validation separate when an explicit test set exists', () => {
  expect(inspectDatasetFolder([...training.map(p => 'train/' + p), 'test/ants/a.png', 'val/bees/b.png']))
    .toMatchObject({ testCount: 1, validationCount: 1, testSource: 'test' });
});
it('rejects unknown test labels and missing test split', () => {
  expect(() => inspectDatasetFolder([...training.map(p => 'train/' + p), 'test/cat/a.png'])).toThrow(/类别/);
  expect(() => inspectDatasetFolder(training.map(p => 'train/' + p))).toThrow(/需要 train/);
});
