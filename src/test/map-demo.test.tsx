import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, expect, it, vi } from 'vitest';
import { MemoryRouter, useLocation } from 'react-router-dom';
import RootApp from '../RootApp';

vi.mock('../platform/PlatformApp', () => ({ default: () => <div>训练平台首页</div> }));
afterEach(cleanup);
function Location() { return <output aria-label="页面路径">{useLocation().pathname}</output>; }
it.each(['/', '/demo', '/demo/legacy'])('routes %s to the workspace without mounting the removed map', async (path) => {
  render(<MemoryRouter initialEntries={[path]}><RootApp /><Location /></MemoryRouter>);
  expect(await screen.findByText('训练平台首页')).toBeInTheDocument();
  expect(screen.getByLabelText('页面路径')).toHaveTextContent(/^\/$/);
  expect(screen.queryByAltText('中国行政区域演示底图')).not.toBeInTheDocument();
});
