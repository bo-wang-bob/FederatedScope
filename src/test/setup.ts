import '@testing-library/jest-dom/vitest';
import { beforeEach, vi } from 'vitest';
// Existing workflow regressions run against the full catalog; scope tests opt in explicitly.
beforeEach(() => vi.stubEnv('VITE_DEMO_SCOPE', 'all'));
