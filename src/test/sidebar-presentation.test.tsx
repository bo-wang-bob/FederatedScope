import { cleanup, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { ConsoleShell, PhotoHome } from '../design/Presentation';
import { PLATFORM_NAME } from '../platform/branding';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('restored sidebar presentation', () => {
  it.each([
    ['jobs', '训练实验'], ['evaluate', '模型验证'], ['backdoorCompare', '后门防御'], ['privacy', '隐私保护'],
  ] as const)('preserves module navigation for %s deep links', (view, label) => {
    const { container } = render(<MemoryRouter><ConsoleShell view={view}>页面内容</ConsoleShell></MemoryRouter>);
    const sidebar = screen.getByRole('complementary');
    expect(within(sidebar).getByRole('link', { name: label })).toHaveAttribute('aria-current', 'page');
    expect(sidebar.querySelectorAll('[aria-current="page"]')).toHaveLength(1);
    expect(within(sidebar).getByRole('link', { name: PLATFORM_NAME + ' · 返回首页' })).toHaveAttribute('href', '/');
    expect(container.querySelector('.command-header')).not.toBeInTheDocument();
  });

  it('restores the three core home actions without requests or a topology map', () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    render(<MemoryRouter><ConsoleShell view="home" mode="live" connected><PhotoHome /></ConsoleShell></MemoryRouter>);
    const home = screen.getByRole('region', { name: '功能导航' });
    expect(within(home).getAllByRole('link')).toHaveLength(3);
    expect(within(home).getByRole('heading', { level: 1 })).toHaveTextContent(PLATFORM_NAME);
    expect(within(home).getByRole('link', { name: /新建训练/ })).toHaveAttribute('href', '/?view=train');
    expect(within(home).getByRole('link', { name: '模型验证' })).toHaveAttribute('href', '/?view=experience');
    expect(within(home).getByRole('link', { name: '算法对比' })).toHaveAttribute('href', '/?view=compare');
    expect(screen.queryByRole('region', { name: '功能拓扑地图' })).not.toBeInTheDocument();
    expect(screen.getByRole('status')).toHaveTextContent('服务已连接');
    expect(fetch).not.toHaveBeenCalled();
  });
});
