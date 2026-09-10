import { act, cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import { MemoryRouter } from 'react-router-dom';
import RootApp from '../RootApp';
import MapDemoPage from '../pages/MapDemoPage';
import { useAppStore } from '../store/useAppStore';

const { platformRendered } = vi.hoisted(() => ({ platformRendered: vi.fn() }));
vi.mock('../platform/PlatformApp', () => ({ default: () => { platformRendered(); return <div>真实训练平台</div>; } }));
beforeAll(() => {
  window.matchMedia = vi.fn().mockImplementation(query => ({ matches: false, media: query,
    addListener: vi.fn(), removeListener: vi.fn(), addEventListener: vi.fn(), removeEventListener: vi.fn() }));
  globalThis.ResizeObserver = class { observe() {} unobserve() {} disconnect() {} };
});
beforeEach(() => {
  vi.clearAllMocks();
  useAppStore.setState({ phaseIndex: 4, round: 18, running: true, selectedNodeId: undefined,
    activeExperimentId: 'preserve-real-session', dataSource: 'backend', connectionState: 'connected' });
});
afterEach(() => { cleanup(); vi.useRealTimers(); vi.unstubAllGlobals(); });

describe('preserved map demo', () => {
  it('keeps the original map but never mounts real controls or requests the API', async () => {
    const fetch = vi.fn(); vi.stubGlobal('fetch', fetch);
    render(<MemoryRouter initialEntries={['/demo']}><RootApp /></MemoryRouter>);
    expect(await screen.findByText('纯前端模拟数据，不会启动真实训练')).toBeInTheDocument();
    expect(screen.getByAltText('中国行政区域演示底图')).toBeInTheDocument();
    expect(screen.getByRole('link', { name: /返回系统首页/ })).toHaveAttribute('href', '/');
    expect(screen.queryByText(/训练数据 · 已连接/)).not.toBeInTheDocument();
    expect(platformRendered).not.toHaveBeenCalled();
    expect(fetch).not.toHaveBeenCalled();
    expect(useAppStore.getState().activeExperimentId).toBe('preserve-real-session');
  }, 20000);
  it('preserves the real platform as the default entry', async () => {
    render(<MemoryRouter initialEntries={['/']}><RootApp /></MemoryRouter>);
    expect(await screen.findByText('真实训练平台')).toBeInTheDocument();
    expect(screen.queryByAltText('中国行政区域演示底图')).not.toBeInTheDocument();
  });
  it('plays, pauses and switches phases locally and clears its timer', () => {
    vi.useFakeTimers();
    const view = render(<MapDemoPage />);
    act(() => vi.advanceTimersByTime(3000));
    expect(useAppStore.getState().phaseIndex).toBe(5);
    fireEvent.click(screen.getByRole('button', { name: /暂停/ }));
    act(() => vi.advanceTimersByTime(6000));
    expect(useAppStore.getState().phaseIndex).toBe(5);
    fireEvent.click(screen.getByRole('button', { name: /中央下发/ }));
    expect(useAppStore.getState().phaseIndex).toBe(0);
    view.unmount();
    act(() => vi.advanceTimersByTime(6000));
    expect(useAppStore.getState().phaseIndex).toBe(0);
  }, 20000);
});
