# Frontend review imagery

Images are bundled locally. Preview pages do not request the platform API or external image hosts. They do not contain model predictions, operational geography, or measured research results.

## Scene image

`namib-landsat.jpg`: [Where the Dunes End](https://science.nasa.gov/earth/earth-observatory/where-the-dunes-end-146064/), NASA Earth Observatory. NASA Earth Observatory image by Joshua Stevens, using Landsat data from the U.S. Geological Survey. Landsat 8 OLI acquisition: 2019-11-13.

- [Original image](https://assets.science.nasa.gov/dynamicimage/assets/science/esd/eo/images/imagerecords/146000/146064/namibia_oli_2019317_lrg.jpg?crop=faces%2Cfocalpoint&fit=clip&h=3618&w=5427)
- [NASA media usage guidance](https://www.nasa.gov/nasa-brand-center/images-and-media/)
- Downloaded 2026-09-15, 1,940,761 bytes.
- SHA-256: `a859053b31ae6268c735e0881743b55ce7192563636486d88ac42d887424ee83`
- Scene illustration only. It is not this system's test data, an operational map, or evidence of NASA endorsement. No NASA logos are used.

## Dataset preview images

`aerial-b52.jpg`, `natural-b52.jpg`, `natural-f16.jpg`, and `natural-c17.jpg` are unmodified copies from `backend/resources/datasets/MilitaryAircraft3D/`, using `0000.jpg` in the matching domain/class directory. SHA-256 values and original labels are recorded in `../assets.ts`.

These images are design-preview assets, not predictions or accuracy evidence. The preview does not use their IDs for backend inference. Original dataset/image rights still apply; no new redistribution license is asserted.

Attribution and scope limitations are retained in this document and `../assets.ts`; the console no longer exposes an image-source button or modal.
