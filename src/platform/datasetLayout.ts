export type FolderLayout = { layout: 'classes' | 'split'; classes: string[]; trainCount: number; testCount: number; validationCount: number; testSource?: string };

export function inspectDatasetFolder(paths: string[]): FolderLayout {
  if (!paths.length) throw new Error('所选文件夹没有支持的图片');
  const rows = paths.map(path => path.split('/'));
  const splitNames = new Set(['train', 'test', 'val', 'valid', 'validation']);
  const split = rows.some(parts => parts.length === 3 && splitNames.has(parts[0].toLowerCase()));
  if (rows.some(parts => parts.length !== (split ? 3 : 2)) || split && rows.some(parts => !splitNames.has(parts[0].toLowerCase()))) {
    throw new Error('请选择训练集最外层文件夹（类别/图片），或完整数据集文件夹（train、test 或 val/类别/图片），不要混合层级。');
  }
  const buckets = new Set(rows.map(parts => parts[0].toLowerCase()));
  const validation = [...buckets].filter(name => ['val', 'valid', 'validation'].includes(name));
  if (split && (!buckets.has('train') || (!buckets.has('test') && !validation.length)))
    throw new Error('完整数据集需要 train 和 test（或 val/valid/validation）文件夹');
  if (split && validation.length > 1) throw new Error('检测到多个验证目录，请只保留 val、valid、validation 中的一个');
  const training = split ? rows.filter(parts => parts[0].toLowerCase() === 'train') : rows;
  const counts = new Map<string, number>();
  training.forEach(parts => { const category = parts[split ? 1 : 0]; counts.set(category, (counts.get(category) || 0) + 1); });
  if (counts.size < 2 || [...counts.values()].some(count => count < 3)) throw new Error('至少 2 个训练类别，每个类别至少 3 张训练图片');
  if (split && rows.some(parts => !counts.has(parts[1]))) throw new Error('测试/验证类别名必须与训练类别名完全一致');
  const testSource = split ? buckets.has('test') ? 'test' : validation[0] : undefined;
  return { layout: split ? 'split' : 'classes', classes: [...counts.keys()].sort(), trainCount: training.length,
    testCount: split ? rows.filter(parts => parts[0].toLowerCase() === testSource).length : 0,
    validationCount: split && buckets.has('test') ? rows.filter(parts => validation.includes(parts[0].toLowerCase())).length : 0,
    testSource };
}
