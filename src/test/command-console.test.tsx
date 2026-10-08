import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import { CommandHome } from '../design/CommandHome';
import { ConsoleShell } from '../design/Presentation';
import type { Catalog, Job, Library } from '../platform/api';

afterEach(() => { cleanup(); vi.unstubAllGlobals(); });

describe('command workspace presentation contract', () => {
  it.each([
    ['jobs', '训练实验'], ['evaluate', '模型验证'], ['backdoorCompare', '后门防御'], ['privacy', '隐私保护'],
  ] as const)('keeps the correct module active for %s deep links', (view, label) => {
    render(<MemoryRouter><ConsoleShell view={view}>content</ConsoleShell></MemoryRouter>);
    const nav = screen.getByRole('navigation', { name: '主要功能' });
    expect(within(nav).getAllByRole('link')).toHaveLength(6);
    expect(within(nav).getByRole('link', { name: label })).toHaveAttribute('aria-current', 'page');
    expect(nav.querySelectorAll('[aria-current="page"]')).toHaveLength(1);
  });

  it('uses only supplied inventory and real job metrics, without fetching or modifying data', () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    const jobs = [
      { id: 'running', action: 'train', status: 'running', request: { name: '实际实验', method: 'fedavg', rounds: 10 }, stage: '第 3 轮', metrics: [{ round: 3 }], createdAt: '2026-10-08' },
      { id: 'done', action: 'train', status: 'completed', request: { name: '已完成实验', method: 'fedprox' }, createdAt: '2026-10-07' },
      { id: 'eval', action: 'evaluate', status: 'completed', request: {}, createdAt: '2026-10-06' },
    ] as Job[];
    const catalog = { groups: [{ dataset: 'A' }, { dataset: 'A' }, { dataset: 'B' }] } as Catalog;
    const library = { models: [{ id: 'model' }], testsets: [{ id: 'test-1' }, { id: 'test-2' }] } as Library;
    render(<MemoryRouter><CommandHome jobs={jobs} catalog={catalog} library={library} /></MemoryRouter>);
    const overview = screen.getByRole('region', { name: '实验概览' });
    expect(within(overview).getByText('完成训练').nextElementSibling).toHaveTextContent('1项');
    expect(within(overview).getByText('数据集').nextElementSibling).toHaveTextContent('2组');
    const inventory = screen.getByRole('region', { name: '模型与测试集' });
    expect(within(inventory).getByText('已保存模型').nextElementSibling).toHaveTextContent('1个');
    expect(within(inventory).getByText('测试集').nextElementSibling).toHaveTextContent('2组');
    expect(screen.getByRole('progressbar')).toHaveAttribute('aria-valuenow', '30');
    expect(screen.getByRole('link', { name: '当前任务 实际实验' })).toHaveAttribute('href', '/?view=jobs&id=running');
    expect(fetch).not.toHaveBeenCalled();
  });

  it('bounds map zoom, supports reset/layers, and does not start work when exploring', () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    const { container } = render(<MemoryRouter><CommandHome preview /></MemoryRouter>);
    const plane = container.querySelector('.command-map-plane');
    const zoomIn = screen.getByRole('button', { name: '放大地图' });
    fireEvent.click(zoomIn); fireEvent.click(zoomIn);
    expect(plane).toHaveAttribute('data-zoom', '1.2'); expect(zoomIn).toBeDisabled();
    fireEvent.click(screen.getByRole('button', { name: '重置地图' }));
    expect(plane).toHaveAttribute('data-zoom', '1.0');
    const zoomOut = screen.getByRole('button', { name: '缩小地图' });
    fireEvent.click(zoomOut); fireEvent.click(zoomOut);
    expect(plane).toHaveAttribute('data-zoom', '0.8'); expect(zoomOut).toBeDisabled();
    const layers = screen.getByRole('button', { name: '显示拓扑连线' });
    fireEvent.click(layers); expect(layers).toHaveAttribute('aria-pressed', 'false');
    expect(container.querySelector('.command-topology-links')).toHaveClass('is-hidden');
    expect(container.querySelectorAll('iframe')).toHaveLength(0);
    expect(container.querySelectorAll('img[src^="http"]')).toHaveLength(0);
    expect(fetch).not.toHaveBeenCalled();
  });

  it('does not turn missing inventory or loading state into invented counts', () => {
    render(<MemoryRouter><CommandHome loading /></MemoryRouter>);
    expect(screen.getAllByText('—')).toHaveLength(4);
    expect(screen.queryByRole('progressbar')).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: '独立评测' })).toHaveAttribute('href', '/?view=evaluate');
    const map = screen.getByRole('region', { name: '功能拓扑地图' });
    expect(within(map).getByRole('link', { name: '隐私保护' })).toHaveAttribute('href', '/?view=privacy');
    expect(within(map).getByRole('link', { name: '后门防御' })).toHaveAttribute('href', '/?view=backdoor');
  });
});
