import { expect, it } from 'vitest';
import { imageExtensions, imageAccept, uploadImages } from '../platform/imageFormats';

it.each(imageExtensions)('accepts %s in training folders and flat tests, including uppercase', extension => {
  const training = new File(['image'], '1.' + extension.toUpperCase());
  Object.defineProperty(training, 'webkitRelativePath', { value: 'Train/Class/1.' + extension.toUpperCase() });
  const test = new File(['image'], '2.' + extension);
  expect(uploadImages([training, test])).toEqual([training, test]);
  expect(imageAccept.split(',')).toContain('.' + extension);
});
it('ignores metadata but does not silently drop unsupported image formats', () => {
  const jpg = new File(['image'], 'photo.jpg');
  expect(uploadImages([jpg, new File(['meta'], '.DS_Store'), new File(['meta'], 'labels.json')])).toEqual([jpg]);
  expect(() => uploadImages([jpg, new File(['raw'], 'photo.cr2')])).toThrow('photo.cr2');
  expect(() => uploadImages([new File(['svg'], 'photo.svg')])).toThrow('photo.svg');
});
