import terrain from './media/namib-landsat.jpg';
import aerialB52 from './media/aerial-b52.jpg';
import naturalC17 from './media/natural-c17.jpg';
import naturalF16 from './media/natural-f16.jpg';
import naturalB52 from './media/natural-b52.jpg';

export const landscape = terrain;
export const imageSource = 'https://science.nasa.gov/earth/earth-observatory/where-the-dunes-end-146064/';
export const testsetId = 'military-design-samples';

// Static, byte-for-byte copies of existing test images. These are not inference results.
export const samples = [
  { id: 'aerial-b52', src: aerialB52, domain: 'aerial', category: 'B-52', label: 'B-52', filename: '0000.jpg', sha256: '62629af7f0a5ca98d902369ab3292204cc59952f911c8e35366d9bccb763737c' },
  { id: 'natural-b52', src: naturalB52, domain: 'natural', category: 'B-52', label: 'B-52', filename: '0000.jpg', sha256: '573c64f0ccef92807b409ef62414d3451c248ce4d313ba0a1eedfa038dd819b0' },
  { id: 'natural-f16', src: naturalF16, domain: 'natural', category: 'F-16', label: 'F-16', filename: '0000.jpg', sha256: '2c90325b781376fb679ebbc1c41b41ff9c97e6ad675fb2bc8b0cb3104cc61a7c' },
  { id: 'natural-c17', src: naturalC17, domain: 'natural', category: 'C-17', label: 'C-17', filename: '0000.jpg', sha256: '9d76f2d9a2270b2b1ab4a6f2f930fed333241829439ccaed51a092969ab50e04' },
] as const;
export type DesignSample = typeof samples[number];
