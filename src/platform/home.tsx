import { CommandHome } from '../design/CommandHome';
import type { Catalog, Job, Library } from './api';

export function SystemHome(props: { jobs: Job[]; loading: boolean; showDatasetImages?: boolean; library?: Library; catalog?: Catalog }) {
  return <CommandHome {...props} />;
}
