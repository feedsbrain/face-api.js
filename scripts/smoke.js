/* eslint-disable */
/**
 * Post-migration smoke test (Node + @tensorflow/tfjs-node).
 *
 * Not a replacement for the full Karma/Jasmine suite (port still pending, see
 * MIGRATION.md). This is a fast, dependency-light regression gate that proves:
 *   - all shipped models load on the current Node / tfjs-node
 *   - face recognition descriptors still match the frozen fixtures numerically
 *   - the tiny face detector still localises the reference faces
 *   - the full detectAllFaces().withFaceLandmarks().withFaceExpressions()
 *     .withAgeAndGender().withFaceDescriptors() pipeline runs end to end
 *
 * Run:  node scripts/smoke.js       (after `npm run build`)
 */
require('@tensorflow/tfjs-node');
const path = require('path');
const fs = require('fs');
const canvas = require('canvas');

const ROOT = path.resolve(__dirname, '..');
const faceapi = require(path.join(ROOT, 'build/commonjs'));

const { Canvas, Image, ImageData } = canvas;
faceapi.env.monkeyPatch({ Canvas, Image, ImageData });

function euclid(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return Math.sqrt(s);
}
function byOrigin(a, b) {
  return Math.hypot(a.x, a.y) - Math.hypot(b.x, b.y);
}

let failures = 0;
function check(name, ok, detail) {
  console.log(`${ok ? '  PASS' : '  FAIL'}  ${name}${detail ? '  — ' + detail : ''}`);
  if (!ok) failures++;
}

(async () => {
  const W = path.join(ROOT, 'weights');
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(W);
  await faceapi.nets.tinyFaceDetector.loadFromDisk(W);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(W);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(W);
  await faceapi.nets.faceExpressionNet.loadFromDisk(W);
  await faceapi.nets.ageGenderNet.loadFromDisk(W);
  console.log(`models loaded, backend = ${faceapi.tf.getBackend()}\n`);

  // 1) numeric golden check against the committed descriptor fixture
  const face1 = await canvas.loadImage(path.join(ROOT, 'test/images/face1.png'));
  const desc1 = await faceapi.nets.faceRecognitionNet.computeFaceDescriptor(
    faceapi.createCanvasFromMedia(face1)
  );
  const expected1 = JSON.parse(
    fs.readFileSync(path.join(ROOT, 'test/data/faceDescriptor1.json'))
  );
  const dist1 = euclid(desc1, expected1);
  check('faceRecognitionNet descriptor matches fixture', desc1.length === 128 && dist1 < 0.1,
    `len=${desc1.length} euclid=${dist1.toFixed(5)} (tol 0.1)`);

  // 2) tinyFaceDetector localisation against the reference boxes
  const facesImg = await canvas.loadImage(path.join(ROOT, 'test/images/faces.jpg'));
  const fc = faceapi.createCanvasFromMedia(facesImg);
  const tiny = (await faceapi.detectAllFaces(fc, new faceapi.TinyFaceDetectorOptions()))
    .map(d => d.box).sort(byOrigin);
  const expectedBoxes = [
    { x: 29, y: 264, width: 139, height: 137 }, { x: 224, y: 240, width: 147, height: 128 },
    { x: 547, y: 81, width: 136, height: 114 }, { x: 214, y: 53, width: 124, height: 119 },
    { x: 430, y: 183, width: 162, height: 143 }, { x: 54, y: 33, width: 134, height: 114 },
  ].sort(byOrigin);
  let maxDelta = 0;
  tiny.forEach((b, i) => {
    const e = expectedBoxes[i];
    if (!e) return;
    maxDelta = Math.max(maxDelta,
      Math.abs(b.x - e.x), Math.abs(b.y - e.y),
      Math.abs(b.width - e.width), Math.abs(b.height - e.height));
  });
  check('tinyFaceDetector finds the 6 reference faces', tiny.length === 6 && maxDelta < 6,
    `count=${tiny.length} maxBoxDelta=${maxDelta.toFixed(1)}px (tol 6)`);

  // 3) full high-level pipeline runs and yields well-formed results
  const full = await faceapi
    .detectAllFaces(fc, new faceapi.SsdMobilenetv1Options())
    .withFaceLandmarks()
    .withFaceExpressions()
    .withAgeAndGender()
    .withFaceDescriptors();
  const wellFormed = full.length >= 5 && full.every(r =>
    r.landmarks.positions.length === 68 &&
    r.descriptor.length === 128 &&
    Number.isFinite(r.age) &&
    (r.gender === 'male' || r.gender === 'female') &&
    Math.abs(r.expressions.asSortedArray().reduce((s, e) => s + e.probability, 0) - 1) < 1e-3);
  check('full SSD pipeline (landmarks + expr + age/gender + descriptors)', wellFormed,
    `${full.length} faces`);

  console.log(`\n${failures === 0 ? 'SMOKE OK' : `SMOKE FAILED (${failures})`}`);
  process.exit(failures === 0 ? 0 : 1);
})().catch(e => { console.error('SMOKE ERROR\n', e); process.exit(1); });
