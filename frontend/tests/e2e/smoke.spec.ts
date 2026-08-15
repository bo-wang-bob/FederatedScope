import { expect, test } from '@playwright/test';

const workspaceRoutes = [
  ['/scenario-analysis', '场景与异构分析'],
  ['/experiments/new', '实验配置与模式编排'],
  ['/experiments/demo/live', '实验运行监控'],
  ['/experiments/demo/privacy', '隐私攻击效果评估'],
  ['/experiments/demo/compare', '攻防与保护效果对照'],
  ['/reports', '实验档案与报告'],
  ['/settings', '系统与演示设置'],
] as const;

test.describe('核心页面', () => {
  for (const [route, heading] of workspaceRoutes) {
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

test('全屏地图、四域、60 客户端和隐藏导航符合首页设计', async ({ page }) => {
  await page.goto('/overview');
  await expect(page.getByRole('heading', { name: '全域信息汇聚 · 跨域智能协同' })).toBeVisible();
  await expect(page.locator('.app-sider')).toHaveCount(0);
  await expect(page.locator('.app-header')).toHaveCount(0);
  await expect(page.locator('.immersive-overview-page')).toHaveCSS('position', 'fixed');
  await expect(page.locator('.map-central-copy b')).toHaveText('中央主服务器');
  for (const domain of ['数字孪生域', '战术符号域', '装备数据库域', '实景侦察域']) {
    await expect(page.getByText(domain, { exact: true }).first()).toBeVisible();
  }
  await expect(page.locator('.map-client-marker')).toHaveCount(60);
  await page.getByRole('button', { name: '打开导航' }).click();
  await expect(page.getByText('场景与异构分析', { exact: true })).toBeVisible();
});

test('训练流程使用高对比双层链路、箭头和真实方向反转', async ({ page }) => {
  await page.goto('/overview');
  await page.getByRole('button', { name: '暂停' }).click();

  await page.locator('.process-step').filter({ hasText: '中央下发' }).click();
  await expect(page.locator('.map-central-backbone')).toHaveCount(4);
  await expect(page.locator('.map-central-link.active')).toHaveCount(4);
  await expect(page.locator('.map-central-link.active').first()).toHaveAttribute('marker-mid', 'url(#map-arrow-cyan)');
  await expect(page.locator('[data-link="central"][data-direction="downlink"]')).toHaveCount(4);
  await expect(page.locator('.map-central-link.active').first()).toHaveCSS('stroke-width', '4px');

  await page.locator('.process-step').filter({ hasText: '域级上传' }).click();
  await expect(page.locator('[data-link="central"][data-direction="uplink"]')).toHaveCount(4);
  await expect(page.locator('.map-central-link.active')).toHaveCount(4);

  await page.locator('.process-step').filter({ hasText: '节点上传' }).click();
  await expect(page.locator('.map-client-link.active')).toHaveCount(60);
  await expect(page.locator('.map-client-link.active').first()).toHaveAttribute('marker-mid', /map-arrow/);
  await expect(page.getByText('24 / 60 已上传')).toBeVisible();
});

test('场景与异构分析只调 alpha 并展示逐客户端分布', async ({ page }) => {
  await page.goto('/scenario-analysis');
  await expect(page.getByText('域间特征偏移为数据集固有属性，只读不可调')).toBeVisible();
  await expect(page.getByText('客户端 × 类别分布热力图')).toBeVisible();
  await expect(page.getByText('客户端分布表')).toBeVisible();
  await expect(page.locator('.scenario-analysis-page .ant-table-row')).toHaveCount(15);
  await expect(page.getByText('OH-DT-C01', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: 'α 0.1' }).click();
  await expect(page.getByRole('button', { name: '确认应用当前预览' })).toBeEnabled();
  await page.getByRole('tab', { name: /战术符号域/ }).click();
  await expect(page.getByText('OH-TS-C01', { exact: true })).toBeVisible();
});

test('旧分析路由统一重定向到合并页面', async ({ page }) => {
  for (const route of ['/scenario', '/heterogeneity']) {
    await page.goto(route);
    await expect(page).toHaveURL(/\/scenario-analysis$/);
    await expect(page.getByRole('heading', { name: '场景与异构分析' })).toBeVisible();
  }
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
