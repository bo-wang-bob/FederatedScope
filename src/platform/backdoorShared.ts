// 后门研究共用常量：三连对比图的顺序与文案，选择与逐样本对照两个页面共用。
export const IMAGE_KEYS = ['clean', 'triggered', 'defense'] as const;
export type ImageKey = typeof IMAGE_KEYS[number];

export const imageMeta: Record<ImageKey, { title: string; caption: string; tone: string }> = {
  clean: { title: '干净样本', caption: '无防御模型 · 未注入触发器', tone: 'baseline' },
  triggered: { title: '注入触发器', caption: '无防御模型 · 触发器已生效', tone: 'attack' },
  defense: { title: '防御后', caption: '防御模型 · 同样注入触发器', tone: 'defense' },
};

export function readable(name: string) { return name.replaceAll('_', ' '); }
