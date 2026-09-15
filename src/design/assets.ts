import terrain from './media/namib-landsat.jpg';
import artRadio from './media/art-radio.jpg';
import realComputer from './media/real-computer.jpg';
import realLaptop from './media/real-laptop.jpg';
import realRadio from './media/real-radio.jpg';

export const landscape = terrain;
export const imageSource = 'https://science.nasa.gov/earth/earth-observatory/where-the-dunes-end-146064/';
export const testsetId = '65f09e544a114f4a9e9eaa00d4d0f0e4';

// Static, byte-for-byte copies of existing test images. These are not inference results.
export const samples = [
  { id: '1cd02b0c02c09840cc4aef22', src: artRadio, domain: 'Art', category: 'Radio', label: '无线电设备', filename: '00045.jpg', sha256: '3fa50a2860d5ab6c02e159deec3c55db90e52c175030ac242ef87f84c292e7bd' },
  { id: 'f6dfa4dd5bcf3f7c176f3431', src: realRadio, domain: 'Real_World', category: 'Radio', label: '无线电设备', filename: '00007.jpg', sha256: '7999d1bfe85282f32b86a5a3c1d4a002ff7099c62742fcc9eae298814162783f' },
  { id: '9098c73c1afa5c755c389d70', src: realLaptop, domain: 'Real_World', category: 'Laptop', label: '笔记本电脑', filename: '00066.jpg', sha256: 'bab042eb3cfd290193e783e0035cc52c97decdb86a685cf440d6cf02205bccf7' },
  { id: '1de59828b76ddc88e1145c35', src: realComputer, domain: 'Real_World', category: 'Computer', label: '计算机', filename: '00020.jpg', sha256: '1a5a932c457709f70af37e256bfb93525bedede67ddeb11d0771079796b34956' },
] as const;
export type DesignSample = typeof samples[number];
