export const imageExtensions = ['jpg', 'jpeg', 'jpe', 'jfif', 'png', 'apng', 'webp', 'bmp', 'dib',
  'tif', 'tiff', 'gif', 'avif', 'heic', 'heif', 'jp2', 'j2k', 'ppm', 'pgm', 'pbm', 'pnm', 'tga'];
export const imageAccept = imageExtensions.map(ext => '.' + ext).join(',');
const extensions = new Set(imageExtensions);
export const supportedImage = (name: string) => extensions.has(name.split('.').at(-1)?.toLowerCase() || '');
const sidecars = /(^|\/)(\.[^/]*|thumbs\.db|desktop\.ini)$|\.(txt|csv|json|xml|yaml|yml|md)$/i;
export function uploadImages(files: File[]) {
  const unsupported = files.filter(file => !supportedImage(file.name) && !sidecars.test(file.webkitRelativePath || file.name));
  if (unsupported.length) throw new Error(`不支持的文件：${unsupported.slice(0, 3).map(file => file.name).join('、')}，请转换为 PNG 或 JPG`);
  return files.filter(file => supportedImage(file.name) && !(file.webkitRelativePath || file.name).split('/').some(part => part.startsWith('.')));
}
