# Frontend design review — 2026-09-15

## Boundary

User instruction: subsequent changes are frontend-first. Backend integration and deployment wait for explicit design approval.

- Worktree: `FederatedScope-worktrees/single-host-platform`.
- Branch: `codex/frontend-design-review`, based on `580937692e5585651a910927854c21825ea28506`.
- Original working tree, server services, data, models and deployed frontend are unchanged.
- No task submission, service restart, server deployment, or remote push is part of this revision.

## Preview

From `frontend`:

```text
npm run dev:design
npm run build:design
```

Preview URL: `http://127.0.0.1:5174/`. The server binds to loopback, uses a separate Vite `design` mode, refuses `/api` requests, and never forwards them to a live backend. `dist-design` is independent of the ordinary `dist` output. Standard `npm run dev` / `build` continue to select the existing live app.

The frontend preview does not mount `PlatformApp`, use the training launcher, poll jobs or inventories, or initiate inference. Form options are explicit design fixtures, not server capability claims. A permanent 14px **设计预览** status identifies this mode.

## Design

- Home: one photographic training entry, model verification and algorithm comparison. No inventories, decorative section numbers, descriptive paragraphs or duplicate research cards.
- Sidebar: four primary areas; map simulation, privacy and backdoor reservations. No group captions, brand subtitle or environment footer.
- Real local imagery: NASA Landsat scene plus four existing Office-Home samples. Credits and provenance limitations are available on demand; see `frontend/src/design/media/README.md`.
- Controls/body: 14px. Section titles: 16px. Page heading: 22px; home heading: 30px. Restrained dark gray/green surfaces and muted green controls.
- Training: reuse current form, validation and parameter logic; remove redundant helper copy in design mode. The preview draft stays in React state, survives module navigation and never overwrites the live localStorage draft. Reload resets this temporary preview state.
- Model verification: filter domain/class, select a real image, previous/next, enlarge; incomplete/broken images cannot enter prediction. Model choices are scheme placeholders, not loaded checkpoints. Clicking prediction reports that no inference ran.
- Evaluation: select scheme and preview domains; empty selection disables submission. No synthetic metrics.
- Comparison: scheme selection and dimension controls; empty state and disabled export until actual results exist.
- `/demo` retains the original, clearly labelled, pure-frontend map simulation. Privacy/backdoor pages contain only a reserved status, no execution controls.
- Hover/focus feedback, short transitions, reduced-motion support. Desktop layout; mobile redesign is out of scope.

## Acceptance

Verified on 2026-09-15:

- All 59 frontend tests passed across 12 files, including 7 new design-preview tests (48.83 seconds).
- Both `npm run build:design` and `npm run build` passed. The design bundle excludes the live `PlatformApp`/chart chunks; the normal build excludes the design module and its new photographs.
- A direct request to the preview's `/api/platform/catalog` returns HTTP 403; there is no backend proxy in design mode.
- Browser checks at 1440×900 and 1280×720: home images, model sample selection, prediction preview dialog, training parameters/actions and comparison empty state. No horizontal overflow at 1280×720. Visible training text has no font sizes below 14px; the action bar remains within the viewport.
- Design tests cover no fetch/XHR requests across navigation/actions, live-draft isolation, form validation and retention, source disclosure, image load/failure gating, filters, evaluation selection, empty comparison and extension routes.
- Existing non-blocking warnings remain in the legacy map/test environment (Drawer deprecation, JSDOM CSS support) and large production chart bundles. The design build has no large-chunk warning.

The user must confirm this frontend design before adapting its pages to real catalog/model/job data. Do not treat the static options, four displayed samples, or button confirmation dialogs as completed backend functionality.
