import { lazy, Suspense } from 'react';
import { Navigate, Route, Routes } from 'react-router-dom';
import { Spin } from 'antd';
import { AppShell } from './components/AppShell';

const OverviewPage = lazy(() => import('./pages/OverviewPage').then((module) => ({ default: module.OverviewPage })));
const ScenarioPage = lazy(() => import('./pages/ScenarioPage').then((module) => ({ default: module.ScenarioPage })));
const HeterogeneityPage = lazy(() => import('./pages/HeterogeneityPage').then((module) => ({ default: module.HeterogeneityPage })));
const ExperimentPage = lazy(() => import('./pages/ExperimentPage').then((module) => ({ default: module.ExperimentPage })));
const LiveMonitorPage = lazy(() => import('./pages/LiveMonitorPage').then((module) => ({ default: module.LiveMonitorPage })));
const PrivacyPage = lazy(() => import('./pages/PrivacyPage').then((module) => ({ default: module.PrivacyPage })));
const ComparePage = lazy(() => import('./pages/ComparePage').then((module) => ({ default: module.ComparePage })));
const ReportsPage = lazy(() => import('./pages/ReportsPage').then((module) => ({ default: module.ReportsPage })));
const SettingsPage = lazy(() => import('./pages/SettingsPage').then((module) => ({ default: module.SettingsPage })));

export default function App() {
  return (
    <AppShell>
      <Suspense fallback={<div className="route-loading"><Spin size="large" /><span>正在加载态势组件…</span></div>}>
        <Routes>
          <Route path="/overview" element={<OverviewPage />} />
          <Route path="/scenario" element={<ScenarioPage />} />
          <Route path="/heterogeneity" element={<HeterogeneityPage />} />
          <Route path="/experiments/new" element={<ExperimentPage />} />
          <Route path="/experiments/:id/live" element={<LiveMonitorPage />} />
          <Route path="/experiments/:id/privacy" element={<PrivacyPage />} />
          <Route path="/experiments/:id/compare" element={<ComparePage />} />
          <Route path="/reports" element={<ReportsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/overview" replace />} />
        </Routes>
      </Suspense>
    </AppShell>
  );
}
