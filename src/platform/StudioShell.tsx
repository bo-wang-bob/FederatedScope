import type { PropsWithChildren } from 'react';
import { ConsoleShell } from '../design/Presentation';
import type { PlatformView } from './navigation';

export function StudioShell(props: PropsWithChildren<{
  view: PlatformView; connected: boolean; running: boolean; openCurrent: () => void;
  showDatasetImages?: boolean;
}>) {
  return <ConsoleShell {...props} mode="live" />;
}
