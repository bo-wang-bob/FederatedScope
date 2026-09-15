# Frontend design review — 2026-09-15

## Design-only boundary (superseded by the approval below)

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

At the end of that revision, design approval was still required. Static options and preview confirmations were not backend functionality.

## Approved live integration — 2026-09-15

The user subsequently requested backend integration and deployment. Work continues in `codex/photo-console-live`, based on the approved design commit `ca60fc6`. The original main working tree and design-only branch remain untouched.

- The live app now shares the approved photographic home, sidebar and source disclosure with the isolated preview. Real training, model, sample, prediction, evaluation and comparison controllers remain in the platform module; design fixtures never supply live form options or results.
- Training retains preflight, parameter submission, idempotency, stop/recovery and server-side cleanup. Model and testset options come from the server. Historical provenance limits, failures and comparability warnings remain visible or available on demand.
- Model verification waits for the selected original image to load before submission. The request includes sample ID and image SHA-256; switching images clears the previous result. Browser QA found and fixed shrinking thumbnail grid rows that made images overlap and selections ambiguous.
- Live development supports `FS_API_PROXY=http://127.0.0.1:18001`. Its reverse proxy preserves Host/Origin so backend same-origin protection remains enabled. The `design` mode still refuses API requests and performs no real work.
- Existing inventory before acceptance: 22 models, 11 testsets and no active tasks. No new training or feature generation was launched.

### Real acceptance

All 61 frontend tests passed across 13 files (`vitest run --pool=threads --maxWorkers=1`), including new live-shell prediction and preflight-to-training payload integration tests. Tests cover the design-only no-request boundary, image load/error gating, routing, draft preservation, launch recovery and comparison guards. Existing JSDOM CSS/pseudo-element and legacy map Drawer warnings remain non-blocking.

Using the existing one-round FedAvg final model `65f09e544a114f4a9e9eaa00d4d0f0e4:final` and its saved Office-Home testset:

- Browser-submitted prediction `8b33130faa004804bed56b232019c049`: Art / Radio / `00045.jpg`; the actual prediction is **Bed**, Softmax 2.6124%, correctly displayed as inconsistent with the true label. Request/result sample ID and image hash match. This is not evidence of model accuracy or improvement.
- Browser-submitted evaluation `60aee9137a3245119caf9faaf2d44e61`: the backend persisted the chosen Art domain and Radio class (47); 11 samples, accuracy 0 and Macro-F1 0. Overall/domain/classification results and confusion matrix are rendered from that result.
- Both jobs completed with `cleanup.ok=true` and no remaining task PIDs. Both JSON exports and the evaluation CSV returned HTTP 200.
- Inference uses the existing frozen feature/classifier workflow, not a new image encoder. Legacy sample-to-feature association limitations remain disclosed.

### Release safety

`frontend/scripts/publish_static.py` accepts only an explicitly checksummed build and Git bundle, a clean expected server base and frontend/docs-only source changes. It validates archive paths, retains all old hashed assets, saves the old index, switches to an independent source branch and atomically replaces the entry page. A failed post-release check restores the previous index/branch. The backend is not restarted and data/model directories are not modified.

Production uses the normal `npm run build` output. `http://127.0.0.1:18001/` is the local tunnel to 4090lziy port 8001; `5174` remains an isolated design preview.
