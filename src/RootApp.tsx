import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { Spin } from 'antd';

const Workspace = lazy(() => import.meta.env.MODE === 'design'
  ? import('./design/DesignPreview') : import('./platform/PlatformApp'));

export default function RootApp() {
  return <Suspense fallback={<div className="route-loading"><Spin /><span>正在加载页面…</span></div>}>
    <Routes>
      <Route path="/demo/*" element={<Navigate to="/" replace />} />
      <Route path="*" element={<Workspace />} />
    </Routes>
  </Suspense>;
}
