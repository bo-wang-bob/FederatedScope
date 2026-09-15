# Desktop Studio redesign

## Scope

Independent worktree/branch: single-host-platform / codex/single-host-platform.
Original feature/GGEUR checkout, backend algorithms, datasets and model files are not modified.

- Four primary areas: home, training, model validation and comparison.
- Horizontal navigation; privacy/backdoor reserved under research extensions. The existing /demo remains isolated.
- No resource/GPU panels, device chooser, cache inventory or manual cache-check action.
- Three-step training wizard; browser-local draft; automatic backend preflight and train.
- Exact request snapshot and separate durable idempotency keys for checking/training.
- Unknown network outcomes must reconcile their original key. Explicit backend rejections and failed checks can be edited. Cancel only targets the owned check before training submission.
- Three-column sample/image/prediction workbench. Model/testset selection travels to independent evaluation.
- Image-load failure disables prediction. Frozen-feature and historical-source limitations remain available and are not misrepresented as end-to-end image inference.
- Unified actual-round comparison curves; protocol mismatch warnings remain visible.
- On-demand charts; no mock operational metrics. Decorative home artwork is CSS, not a topology/status visualization.
- Desktop 1180 px minimum; short desktop viewports keep training actions visible with a scrollable parameter area. Reduced-motion preferences are respected.

## Verification

Run in frontend:

    npm run build
    node node_modules/vitest/vitest.mjs run --maxWorkers=2 --reporter=dot

Tests cover wizard generation/configuration binding, cross-page draft preservation, duplicate submission, interrupted submission, explicit rejection, cancellation, model/testset deep links, real prediction/error rendering, image load failure, operational-navigation removal, and legacy map isolation.

## Release

Serve a built frontend from the existing single-host service on 4090lziy.
Verify the expected remote commit and clean worktree, bundle/archive hashes, and safe archive paths.
Preserve old hashed assets and back up the previous index. Replace the new index atomically after copying assets.
No backend restart, process/port reassignment, dataset relocation, or broad cleanup is required.
