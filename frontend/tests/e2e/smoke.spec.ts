import { expect, test } from '@playwright/test';

const workspaceRoutes = [
  ['/scenario-analysis', '场景与异构分析'],
  ['/experiments/new', '实验配置'],
  ['/experiments/current/live', '实验运行监控'],
  ['/reports', '实验记录'],
] as const;

test.describe('下一阶段核心页面', () => {
  for (const [route, heading] of workspaceRoutes) {
    test(`${heading} 可渲染`, async ({ page }) => {
      await page.goto(route);
      await expect(page.getByRole('heading', { name: heading })).toBeVisible();
      await expect(page.locator('.app-sider')).toBeVisible();
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
  await expect(page.getByText('隐私攻击效果', { exact: true })).toHaveCount(0);
  await expect(page.getByText('对照分析', { exact: true })).toHaveCount(0);
  await expect(page.getByText('系统设置', { exact: true })).toHaveCount(0);
});

test('训练流程使用高对比双层链路、箭头和真实方向反转', async ({ page }) => {
  await page.goto('/overview');
  await page.getByRole('button', { name: '暂停' }).click();
  await page.locator('.process-step').filter({ hasText: '中央下发' }).click();
  await expect(page.locator('.map-central-backbone')).toHaveCount(4);
  await expect(page.locator('.map-central-link.active')).toHaveCount(4);
  await expect(page.locator('.map-central-link.active').first()).toHaveAttribute('marker-mid', 'url(#map-arrow-cyan)');
  await expect(page.locator('[data-link="central"][data-direction="downlink"]')).toHaveCount(4);
  await page.locator('.process-step').filter({ hasText: '域级上传' }).click();
  await expect(page.locator('[data-link="central"][data-direction="uplink"]')).toHaveCount(4);
  await page.locator('.process-step').filter({ hasText: '节点上传' }).click();
  await expect(page.locator('.map-client-link.active')).toHaveCount(60);
  await expect(page.locator('.map-client-link.active').first()).toHaveAttribute('marker-mid', /map-arrow/);
});

test('场景页只设置基础环境并展示逐客户端分布', async ({ page }) => {
  await page.goto('/scenario-analysis');
  await expect(page.getByText('四域特征空间投影')).toHaveCount(0);
  await expect(page.getByText('域间特征距离矩阵')).toHaveCount(0);
  await expect(page.getByText('客户端 × 类别分布热力图')).toBeVisible();
  await expect(page.getByText('客户端分布表')).toBeVisible();
  await expect(page.locator('.scenario-analysis-page .ant-table-row')).toHaveCount(15);
  await page.getByRole('button', { name: 'α 0.1' }).click();
  await page.getByRole('button', { name: '应用到实验配置' }).click();
  await expect(page.getByText('场景已应用')).toBeVisible();
  await page.getByRole('tab', { name: /战术符号域/ }).click();
  await expect(page.getByText('OH-TS-C01', { exact: true })).toBeVisible();
});

test('旧页面路由按下一阶段边界重定向', async ({ page }) => {
  await page.goto('/scenario');
  await expect(page).toHaveURL(/\/scenario-analysis$/);
  await page.goto('/experiments/demo/privacy');
  await expect(page).toHaveURL(/\/experiments\/new$/);
  await page.goto('/experiments/demo/compare');
  await expect(page).toHaveURL(/\/reports$/);
  await page.goto('/settings');
  await expect(page).toHaveURL(/\/overview$/);
});

test('实验模式使用互斥的单一配置块', async ({ page }) => {
  await page.goto('/experiments/new');
  await page.getByRole('button', { name: /隐私保护实验/ }).click();
  await expect(page.getByText('隐私攻击与保护')).toBeVisible();
  await expect(page.getByText('后门攻击与防御')).toHaveCount(0);
  await page.getByRole('button', { name: /后门攻防实验/ }).click();
  await expect(page.getByText('后门攻击与防御')).toBeVisible();
  await expect(page.getByText('隐私攻击与保护')).toHaveCount(0);
});

test('启动成功后使用后端实验编号跳转监控', async ({ page }) => {
  await page.route('**/api/capabilities', async (route) => route.fulfill({ json: { data: { apiVersion: '1.0', datasets: [], devices: ['cpu'], methods: ['fedavg', 'fedprox', 'heterogeneous_solution'], experimentTypes: ['heterogeneity', 'privacy', 'backdoor'], runner: { ready: true, dataReady: true, modelReady: true, templatesReady: true } } } }));
  await page.route('**/api/experiments/preflight', async (route) => route.fulfill({ json: { data: { ready: 'true' } } }));
  await page.route('**/api/experiments', async (route) => {
    if (route.request().method() === 'POST') {
      const config = route.request().postDataJSON();
      await route.fulfill({ status: 201, json: { data: { experimentId: 'EXP-E2E-001', name: config.name, type: config.type, method: config.common.method, scenarioId: config.scenarioId, scenarioSummary: { dataset: 'office-home', alpha: 0.3, seed: 20260815, partitionVersion: 'e2e' }, config, status: 'queued', createdAt: new Date().toISOString(), updatedAt: new Date().toISOString(), sequence: 0, round: 0, totalRounds: config.common.rounds, metrics: [], finalMetrics: {} } } });
    } else {
      await route.fulfill({ json: { data: [] } });
    }
  });
  await page.goto('/scenario-analysis');
  await page.getByRole('button', { name: '应用到实验配置' }).click();
  await expect(page.getByText('场景已应用')).toBeVisible();
  await page.getByText('实验配置', { exact: true }).click();
  await page.getByRole('button', { name: '创建并启动实验' }).click();
  await expect(page).toHaveURL(/\/experiments\/EXP-E2E-001\/live$/);
});
