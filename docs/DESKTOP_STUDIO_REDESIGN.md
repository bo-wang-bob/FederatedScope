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

## Acceptance — 2026-09-15

- Production build passed; 50 tests across 10 files passed. Existing legacy Drawer deprecation/JSDOM CSS-parser warnings remain non-blocking.
- Browser inspected at 1280×720: no horizontal overflow; wizard action remained within the viewport. Long parameter content scrolls independently.
- Main Studio feature chunk reduced from about 1.79 MB to 642 KB uncompressed; the 1.15 MB chart chunk is now deferred until a chart is used. Vite still reports large-chunk warnings.
- A real browser click automatically created preflight a9da13b6839b432d89d0c4d73619c783 and training 65f09e544a114f4a9e9eaa00d4d0f0e4.
- Run configuration: Office-Home/VIT, FedAvg, 1 communication round, 60 logical clients, 4 participating, 4 samples per client, batch 8, local epochs 1, learning rate 0.0001, seed/split seed 42, alpha 0.1.
- Final/best checkpoints saved. The test and partition fingerprints and final accuracy exactly matched prior run 1c7d10c860194eefb805d386ad2b03c6.
- Model validation opened the exact new final checkpoint and its testset. Prediction 8c509c89565740c0a91d785cc9143682 showed Calculator → Screwdriver, explicitly marked incorrect, without substituting the true label.
- Independent evaluation 497254a1571644fd8b7551c8d714ad38 completed on 4,679 samples across Art, Clipart, Product and Real_World. Accuracy 0.013891857234451806 matched the training result; Macro-F1 0.005677097941101542.
- Comparison UI selected both real runs, showed matching protocols and matching metrics on a common actual-round axis.
- JSON, model and evaluation CSV endpoints returned HTTP 200. CSV: 10,471 bytes; model: 135,606 bytes.
- All acceptance jobs ended, with cleanup.ok=true and no remaining owned PIDs. No network training port was allocated.
- Short runs verify wiring/reproducibility, NOT stable method improvement or target accuracy.
- Existing platform/backend PIDs 67386 and legacy 188 were preserved during the frontend release. Original worktree/data were not changed; old hashed assets and an index rollback copy remain.
