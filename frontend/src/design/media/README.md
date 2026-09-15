# Frontend review imagery

Images are bundled locally. Preview pages do not request the platform API or external image hosts. They do not contain model predictions, operational geography, or measured research results.

## Scene image

`namib-landsat.jpg`: [Where the Dunes End](https://science.nasa.gov/earth/earth-observatory/where-the-dunes-end-146064/), NASA Earth Observatory. NASA Earth Observatory image by Joshua Stevens, using Landsat data from the U.S. Geological Survey. Landsat 8 OLI acquisition: 2019-11-13.

- [Original image](https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/images/imagerecords/146000/146064/namibia_oli_2019317_lrg.jpg?crop=faces%2Cfocalpoint&fit=clip&h=3618&w=5427)
- [NASA media usage guidance](https://www.nasa.gov/nasa-brand-center/images-and-media/)
- Downloaded 2026-09-15, 1,940,761 bytes.
- SHA-256: `a859053b31ae6268c735e0881743b55ce7192563636486d88ac42d887424ee83`
- Scene illustration only. It is not this system's test data, an operational map, or evidence of NASA endorsement. No NASA logos are used.

## Existing test images

`art-radio.jpg`, `real-radio.jpg`, `real-laptop.jpg`, `real-computer.jpg` are unchanged copies of four images already in the user's Office-Home testset `65f09e544a114f4a9e9eaa00d4d0f0e4`. Sample IDs, original filenames, original domains/classes and verified SHA-256 values are recorded in `../assets.ts`. They were retrieved read-only on 2026-09-15 and each file hash matched its sample manifest.

The `Art` domain label is preserved even when the source file itself is a photograph. These are generic research classification samples, not military datasets. Their inclusion does not imply that the model is usable for military applications.

Historical feature-to-image correspondence remains source-limited (`legacy-split-seed-ordered-labels; no embedded sample IDs`). Exact image identity does not prove the source of historical feature generation. No inference is performed by this preview. Copies remain subject to the original dataset/image rights; no new license or public redistribution permission is asserted.

Attribution and scope limitations are available from the preview's **图片来源** button, not repeated as persistent page microcopy.
