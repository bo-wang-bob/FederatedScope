import { expect, test } from '@playwright/test';

const routes = [
  ['/overview', '全域信息汇聚 · 跨域智能协同'],
  ['/scenario', '跨域场景编排'],
  ['/heterogeneity', '跨域异构态势分析'],
  ['/experiments/new', '实验配置与模式编排'],
  ['/experiments/demo/live', '实验运行监控'],
  ['/experiments/demo/privacy', '隐私攻击效果评估'],
  ['/experiments/demo/compare', '攻防与保护效果对照'],
  ['/reports', '实验档案与报告'],
  ['/settings', '系统与演示设置'],
] as const;

test.describe('核心页面', () => {
  for (const [route, heading] of routes) {
    test(`${heading} 可渲染`, async ({ page }) => {
      const errors: string[] = [];
      page.on('pageerror', (error) => errors.push(error.message));
      await page.goto(route);
      await expect(page.getByRole('heading', { name: heading })).toBeVisible();
      await expect(page.locator('.app-sider')).toBeVisible();
      expect(errors).toEqual([]);
    });
  }
});

test('三级拓扑、节点状态和真值开关可交互', async ({ page }) => {
  await page.goto('/overview');
  await expect(page.getByText('中央服务器', { exact: true })).toBeVisible();
  await expect(page.getByText('态势感知域', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('D01-N004', { exact: true }).first()).toBeVisible();
  await expect(page.getByText('恶意', { exact: true }).first()).toBeVisible();
  await page.getByRole('switch').first().click();
  await expect(page.locator('.truth-flag')).toHaveCount(0);
});

test('三类隐私攻击效果可分别切换', async ({ page }) => {
  await page.goto('/experiments/demo/privacy');
  await expect(page.getByText('成员得分分布')).toBeVisible();
  await page.getByText('属性推断', { exact: true }).click();
  await expect(page.getByText('合成属性混淆矩阵')).toBeVisible();
  await page.getByText('数据重建', { exact: true }).click();
  await expect(page.getByText('合成参考特征')).toBeVisible();
  await expect(page.getByText('启用保护后')).toBeVisible();
});

test('隐私和后门实验模式不会同时选中', async ({ page }) => {
  await page.goto('/experiments/new');
  await page.getByRole('button', { name: /隐私风险与保护/ }).click();
  await expect(page.getByText('隐私攻击分别评估')).toBeVisible();
  await page.getByRole('button', { name: /后门攻防模拟/ }).click();
  await expect(page.getByText('后门模式已自动关闭隐私攻击与隐私保护')).toBeVisible();
});
