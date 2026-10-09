// 后门研究共用常量：三连对比图的顺序与文案，选择与逐样本对照两个页面共用。
export const IMAGE_KEYS = ['clean', 'triggered', 'defense'] as const;
export type ImageKey = typeof IMAGE_KEYS[number];

export const imageMeta: Record<ImageKey, { title: string; caption: string; tone: string }> = {
  clean: { title: '干净样本', caption: '无防御模型 · 未注入触发器', tone: 'baseline' },
  triggered: { title: '注入触发器', caption: '无防御模型 · 注入该实验触发器', tone: 'attack' },
  defense: { title: '防御后', caption: '防御模型 · 注入该实验触发器', tone: 'defense' },
};

// 无触发器攻击 (label_flip: 数据投毒 / 模型投毒) 没有可注入的触发器, 三格改为
// 「攻击模型干净图 / 无触发器 / 防御模型干净图」——后端的 defenseClean 图在这里
// 显示为第三格, 因为两个模型看的是同一批干净图。
export const triggerlessImageMeta: Record<ImageKey, { title: string; caption: string; tone: string }> = {
  clean: { title: '攻击模型', caption: '攻击模型 · 干净测试图（命中目标类的样本标红）', tone: 'baseline' },
  triggered: { title: '注入触发器', caption: '该攻击不产生触发器，无可注入的对照图', tone: 'attack' },
  defense: { title: '防御后', caption: '防御模型 · 同一批干净测试图', tone: 'defense' },
};

export const metaFor = (triggerless?: boolean) =>
  triggerless ? triggerlessImageMeta : imageMeta;

export function readable(name: string) { return name.replaceAll('_', ' '); }
