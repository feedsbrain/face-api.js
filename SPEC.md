# face-api.js — Project Specification

> Status: living document. Captures what the project is, how it is built, and
> where future development should go. Written against `v0.22.2` (branch `master`).

---

## 1. Purpose

face-api.js is a **JavaScript library for face detection, face recognition,
facial landmark detection, expression recognition, and age/gender estimation**.
It runs the same API in two environments:

- **Browser** — via a UMD bundle (`dist/face-api.js`) or an npm ES6/CommonJS build.
- **Node.js** — with browser primitives (`HTMLImageElement`, `HTMLCanvasElement`,
  `ImageData`) polyfilled, typically by the `canvas` package.

All inference is performed client-side/on-device on top of
**`@tensorflow/tfjs-core`** (pinned to `1.7.0`). No server, no cloud API, no
network calls except to fetch model weight files that the consumer hosts.

### Design goals

1. **One API, two runtimes.** Environment differences are isolated behind
   `src/env/`.
2. **Pretrained, ready to use.** Weights ship in `weights/` and are loaded by
   name; users are not expected to train.
3. **Composable high-level API.** `detectAllFaces(input).withFaceLandmarks().withFaceDescriptors()`
   reads as a pipeline and each stage is optional.
4. **Small models suitable for realtime.** Tiny variants exist for webcam / mobile
   use; larger models exist for accuracy.
5. **No heavy runtime deps.** Only `@tensorflow/tfjs-core` and `tslib`.

---

## 2. High-level architecture

```
                 input (img | video | canvas | tensor | id | Buffer)
                                     │
                          src/dom/toNetInput.ts  ──►  NetInput
                                     │
         ┌───────────────────────────┼─────────────────────────────┐
         ▼                           ▼                             ▼
  Face detectors            Landmark detectors            Face processors
  (locateFaces)             (detectLandmarks)             (predict*)
  ─────────────             ─────────────────             ───────────────
  SsdMobilenetv1            FaceLandmark68Net             FaceRecognitionNet  → 128-d descriptor
  TinyFaceDetector         FaceLandmark68TinyNet         FaceExpressionNet   → 7 expressions
  TinyYolov2 (legacy)                                     AgeGenderNet        → age + gender
  Mtcnn (legacy)  ──► detection + 5-pt landmarks
         │
         ▼
  FaceDetection[] ──► extractFaces / align ──► face chips fed to processors
         │
         ▼
  Composable task results (WithFaceDetection<WithFaceLandmarks<WithFaceDescriptor<…>>>)
         │
         ▼
  src/draw/*  (overlay rendering)   +   FaceMatcher (recognition by descriptor distance)
```

### Core building blocks

| Concern | Location | Notes |
|---|---|---|
| Abstract model base | `src/NeuralNetwork.ts` | load / dispose / serialize params, weight-map vs. flat-`Float32Array` loading, `variable()`/`freeze()` for fine-tuning |
| Environment abstraction | `src/env/` | `isBrowser`, `isNodejs`, `createBrowserEnv`, `createNodejsEnv`, `createFileSystem`, `monkeyPatch` |
| Input normalization | `src/dom/` | `NetInput`, `toNetInput`, `bufferToImage`, `fetchImage/Json`, `extractFaces`, `extractFaceTensors`, `matchDimensions` |
| Common NN layers | `src/common/` | conv layer, depthwise separable conv, fully-connected layer, param extraction factories |
| Tensor ops | `src/ops/` | `nonMaxSuppression`, `iou`, `normalize`, `padToSquare`, `minBbox`, `shuffleArray` |
| Geometry / result classes | `src/classes/` | `Box`, `BoundingBox`, `Rect`, `Point`, `Dimensions`, `FaceDetection`, `FaceLandmarks(5/68)`, `LabeledFaceDescriptors`, `FaceMatch`, `ObjectDetection` |
| Result mixins | `src/factories/` | `WithFaceDetection`, `WithFaceLandmarks`, `WithFaceDescriptor`, `WithFaceExpressions`, `WithAge`, `WithGender` |
| High-level / global API | `src/globalApi/` | `nets` singletons, `detectSingleFace`/`detectAllFaces`, composable `*Task` classes, `FaceMatcher`, `allFaces` (legacy) |
| Drawing | `src/draw/` | `drawDetections`, `DrawBox`, `DrawFaceLandmarks`, `drawFaceExpressions`, `DrawTextField`, `drawContour` |
| Shared feature extractors | `src/faceFeatureExtractor/`, `src/faceProcessor/`, `src/xception/` | backbone reused by expression net and age/gender net |

---

## 3. Models

Weight files live in `weights/` as a `*-weights_manifest.json` + one or more
`*-shardN` binary blobs. They are loaded by directory URL; the manifest name is
derived from `getDefaultModelName()` on each net.

| Net (in `faceapi.nets`) | Task | Output | Model files | Status |
|---|---|---|---|---|
| `ssdMobilenetv1` | Face detection (default) | `FaceDetection[]` (box + score) | `ssd_mobilenetv1_model-*` (~5.4 MB) | Recommended for accuracy |
| `tinyFaceDetector` | Face detection, realtime | `FaceDetection[]` | `tiny_face_detector_model-*` (~190 KB) | Recommended for webcam / mobile |
| `faceLandmark68Net` | 68-point landmarks | `FaceLandmarks68` | `face_landmark_68_model-*` (~350 KB) | Current |
| `faceLandmark68TinyNet` | 68-point landmarks, smaller | `FaceLandmarks68` | `face_landmark_68_tiny_model-*` (~80 KB) | Current |
| `faceRecognitionNet` | Face embedding (ResNet-34-like) | `Float32Array` 128-d descriptor | `face_recognition_model-*` (~6.2 MB) | Current |
| `faceExpressionNet` | Expression classification | `FaceExpressions` (neutral, happy, sad, angry, fearful, disgusted, surprised) | `face_expression_model-*` (~330 KB) | Current |
| `ageGenderNet` | Age regression + gender classification | `{ age, gender, genderProbability }` | `age_gender_model-*` (~420 KB) | Current |
| `mtcnn` | Detection + 5-pt landmarks (single model) | `WithFaceLandmarks<WithFaceDetection>[]` | `mtcnn_model-*` | **Legacy / deprecated**, kept for back-compat |
| `tinyYolov2` | Face detection | `FaceDetection[]` | not shipped in `weights/` | **Legacy / deprecated** |

### Face detector options

- `SsdMobilenetv1Options({ minConfidence = 0.5, maxResults = 100 })`
- `TinyFaceDetectorOptions({ inputSize = 416, scoreThreshold = 0.5 })` — `inputSize`
  must be divisible by 32 (128/160/224/320/416/512/608).
- `MtcnnOptions(...)` — legacy.
- `TinyYolov2Options(...)` — legacy.

---

## 4. Public API surface

Everything is re-exported from `src/index.ts` onto the `faceapi` namespace.

### High-level (recommended)

```js
faceapi.detectSingleFace(input, detectorOptions?)   // → DetectSingleFaceTask
faceapi.detectAllFaces(input, detectorOptions?)     // → DetectAllFacesTask
```

Composable chain methods (each returns a task that is `await`-able and further
chainable):

```
.withFaceLandmarks(useTinyModel?)
.withFaceDescriptor()   / .withFaceDescriptors()
.withFaceExpressions()
.withAgeAndGender()
```

Result shape grows by mixin, e.g.
`WithAge<WithGender<WithFaceDescriptor<WithFaceLandmarks<WithFaceDetection<{}>>>>>`.

### Low-level (direct forward passes)

```js
faceapi.ssdMobilenetv1(input, options)
faceapi.tinyFaceDetector(input, options)
faceapi.detectFaceLandmarks(faceImage) / faceapi.detectFaceLandmarksTiny(faceImage)
faceapi.computeFaceDescriptor(alignedFaceImage)
faceapi.recognizeFaceExpressions(faceImage)
faceapi.predictAgeAndGender(faceImage)
```

### Model loading

```js
await faceapi.nets.ssdMobilenetv1.loadFromUri('/models')      // browser
await faceapi.nets.ssdMobilenetv1.loadFromDisk('./weights')   // node
await faceapi.nets.ssdMobilenetv1.load(float32ArrayOrUrl)
// convenience: faceapi.loadSsdMobilenetv1Model(url), loadFaceLandmarkModel(url), …
```

### Recognition

```js
const matcher = new faceapi.FaceMatcher(labeledDescriptorsOrResults, distanceThreshold = 0.6)
matcher.findBestMatch(descriptor)      // → FaceMatch { label, distance }
faceapi.euclideanDistance(a, b)
new faceapi.LabeledFaceDescriptors(label, [Float32Array, …])
```

### Display helpers

```js
faceapi.matchDimensions(canvas, displaySize)
faceapi.resizeResults(results, displaySize)
faceapi.draw.drawDetections(canvas, results)
faceapi.draw.drawFaceLandmarks(canvas, results)
faceapi.draw.drawFaceExpressions(canvas, results, minProbability)
new faceapi.draw.DrawBox(box, options).draw(canvas)
new faceapi.draw.DrawTextField(lines, anchor, options).draw(canvas)
```

### Environment

```js
faceapi.env.monkeyPatch({ Canvas, Image, ImageData })   // node setup
faceapi.env.isBrowser() / faceapi.env.isNodejs()
faceapi.tf   // re-exported @tensorflow/tfjs-core
```

### Backward-compat aliases (do not remove without a major bump)

`loadFaceDetectionModel`, `locateFaces`, `detectLandmarks`, `allFaces`, `Mtcnn`,
`TinyYolov2`.

---

## 5. Build, test, tooling

| Task | Command | Output |
|---|---|---|
| UMD bundle | `npm run rollup` / `npm run rollup-min` | `dist/face-api.js`, `dist/face-api.min.js` |
| CommonJS build | `npm run tsc` | `build/commonjs/` (`main`, `typings`) |
| ES6 build | `npm run tsc-es6` | `build/es6/` (`module`) |
| Full build | `npm run build` | clears `build/` + `dist/`, runs all four |
| Docs | `npm run docs` | TypeDoc from `src/` |
| Browser tests | `npm run test-browser` | Karma + Jasmine + Chrome (`karma.conf.js`) |
| Node tests | `npm run test-node` | `ts-node` + Jasmine (`jasmine-node.js`, `test/env.node.ts`) |
| Focused browser tests | `npm run test-<net>` | sets `UUT=<net>` env var |
| CPU backend tests | `npm run test-cpu` | sets `BACKEND_CPU=true` |

- **Language:** TypeScript `~3.8`, target `es5`, `module` commonjs (es6 for the
  es6 build). `strictNullChecks` on, `noImplicitAny` on.
- **Bundler:** Rollup 2 with `typescript2`, `node-resolve`, `commonjs`, `uglify`
  (min only). `crypto` is marked external; circular-dependency warnings are
  intentionally silenced.
- **CI:** `.travis.yml` — Node 8/10/11/12 × `{browser, node}`, `BACKEND_CPU=true`,
  `EXCLUDE_UNCOMPRESSED=true`, plus `npm run build`.
- **Test assets:** images/video/JSON fixtures in `test/`, `test/data/*.json` hold
  expected landmark positions / descriptors; `test/tests-legacy/` covers
  uncompressed weight variants.

### Repo layout

```
src/            library source (see §2 table)
weights/        shipped pretrained model manifests + shards
dist/           prebuilt UMD bundles (checked in)
examples/
  examples-browser/   Express + webpack demo app (npm start → :3000)
  examples-nodejs/     ts-node scripts (faceDetection.ts, faceRecognition.ts, …)
  images/ media/        shared demo assets
test/           Karma + Jasmine specs, fixtures, expected outputs
```

---

## 6. Key invariants & gotchas for contributors

1. **Environment init order.** `src/env/index.ts` checks `isBrowser()` *before*
   `isNodejs()` because Electron renderer satisfies both. Do not reorder without
   understanding #599 (electron env fix) and #584.
2. **tfjs-core is pinned exactly (`1.7.0`).** tfjs makes breaking changes at minor
   versions. Any upgrade is a coordinated effort (op signatures, weight loading,
   backends) and must be validated against both test suites.
3. **Weights loaded by convention.** `loadFromUri(dir)` expects
   `<dir>/<modelName>-weights_manifest.json` + shards next to it. Renaming files
   in `weights/` breaks consumers.
4. **Two weight formats.** `NeuralNetwork.load` accepts either a flat
   `Float32Array` (`extractParams`) or a tfjs weight map (`extractParamsFromWeigthMap`).
   Both paths must be kept in sync when a model's architecture changes.
   (Note the intentional legacy misspelling `WeigthMap` throughout the codebase.)
5. **Tensor memory.** Forward passes run inside `tf.tidy`/manual dispose. New code
   that creates tensors outside a net must dispose them or wrap in `tf.tidy`.
6. **Alignment matters for accuracy.** Descriptors/expressions/age-gender are more
   stable when `.withFaceLandmarks()` runs first (provides `alignedRect`). Skipping
   it is supported but documented as "less stable".
7. **`dist/` is committed.** Rebuild and commit it when releasing.
8. **Composable tasks are lazy + thenable.** `*Task` classes extend
   `ComposableTask<T>` (`then`/`run`); returning a plain Promise instead would
   break chaining.
9. **Upstream base.** Draw utilities and some helpers mirror
   `tfjs-image-recognition-base` (same author); historically this repo absorbed
   that code. Keep behavior compatible.

---

## 7. Known limitations / current state

- **Unmaintained upstream.** The original repo (`justadudewhohacks/face-api.js`)
  has been dormant since ~2020 (v0.22.2). This fork tracks small fixes
  (see recent commits: electron env, tfjs-core 1.7.0).
- **Old tfjs.** `tfjs-core@1.7.0` predates the modern `@tensorflow/tfjs` 3.x/4.x
  line; no WebGPU backend, older WASM/WebGL backends only.
- **Node build friction.** Requires `canvas` (native build) and, for speed,
  `@tensorflow/tfjs-node@1.7.0` (native, Python toolchain).
- **`mtcnn` and `tinyYolov2` are legacy** and effectively deprecated; `tinyYolov2`
  weights are not even shipped.
- **Node 6 unsupported**; CI matrix tops out at Node 12.
- No TypeScript `strict` mode (only a subset of strict flags).

---

## 8. Future development directions

Roughly ordered by leverage. None are committed; this is a menu.

### 8.1 Dependency & runtime modernization (highest impact, highest risk)

- [ ] Upgrade `@tensorflow/tfjs-core` 1.7 → 4.x, migrate to the umbrella
  `@tensorflow/tfjs` package, and adopt `@tensorflow/tfjs-node` 4.x.
- [ ] Add a **WASM backend** path and evaluate **WebGPU** for the browser.
- [ ] Replace deprecated tfjs ops; re-verify NMS, `padToSquare`, `normalize`.
- [ ] Regression-test all shipped weights against the new runtime (numerical
  tolerance in `test/data/*.json`).

### 8.2 Build & packaging

- [ ] Move from Rollup 2 + `rollup-plugin-typescript2` to a current toolchain
  (Rollup 4 / esbuild / tsup); emit ESM + CJS + `.d.ts` cleanly.
- [ ] Add `exports` map to `package.json`; stop shipping `dist/` in git (publish
  as release artifacts instead).
- [ ] Upgrade TypeScript (5.x) and turn on full `strict`.
- [ ] Provide a separate lightweight "browser realtime" entry
  (tinyFaceDetector + tinyLandmarks only) for smaller bundles / tree-shaking.

### 8.3 CI & quality

- [ ] Replace Travis with GitHub Actions; matrix on Node 18/20/22, headless
  Chrome + Firefox.
- [ ] Add lint (ESLint + `@typescript-eslint`) and Prettier; no config exists today.
- [ ] Add coverage reporting and a bundle-size check.
- [ ] Deterministic model-download step for CI instead of relying on `weights/`.

### 8.4 Models

- [ ] Retrain / swap the face recognition backbone for a smaller, more accurate
  embedding model; keep the 128-d output contract or version it.
- [ ] Ship a modern single-stage detector (e.g. BlazeFace / YOLOv5-face class)
  behind the existing `FaceDetectionOptions` abstraction.
- [ ] Add optional **face-mesh / 468-point** landmarks as a new net alongside the
  68-point ones.
- [ ] Quantized (int8) weight variants for all nets; expose an
  `EXCLUDE_UNCOMPRESSED`-style switch to consumers.
- [ ] Formally remove `mtcnn` / `tinyYolov2` in a `1.0` major (leave shims that
  throw a helpful error).

### 8.5 API & DX

- [ ] First-class async model-manager: `faceapi.loadModels('/models', ['ssd','landmark68'])`.
- [ ] Typed, discriminated result unions instead of intersection-mixin types
  (better editor autocomplete).
- [ ] Streaming/video helper: a `FaceTracker` that debounces detections across
  frames and reuses tensors.
- [ ] Web Worker / OffscreenCanvas support and an official example.
- [ ] React / Vue / Svelte example rewrites; the current `examples-browser` uses
  an old webpack + Express setup.
- [ ] Node ESM example without `canvas` (construct tensors directly).

### 8.6 Documentation

- [ ] Regenerate TypeDoc and host it; the README links to a stale docs site.
- [ ] "Choosing a detector" and "performance tuning" guides with real numbers on
  current hardware/backends.
- [ ] Migration guide for the tfjs upgrade.

---

## 9. Versioning & compatibility policy (proposed)

- Follow **SemVer** strictly once past `1.0`.
- **Breaking** = removing/renaming a `faceapi.*` export, changing a model's output
  shape or numeric contract, renaming files in `weights/`, or bumping the peer
  tfjs major.
- Keep the §4 back-compat aliases until a major bump; document deprecations one
  minor version before removal.
- Model weights are versioned with the package; if a model's architecture changes,
  publish new manifest filenames rather than overwriting.

---

## 10. Quick reference — end-to-end example

```js
import * as faceapi from 'face-api.js';

await faceapi.nets.ssdMobilenetv1.loadFromUri('/models');
await faceapi.nets.faceLandmark68Net.loadFromUri('/models');
await faceapi.nets.faceRecognitionNet.loadFromUri('/models');

const input = document.getElementById('img');
const results = await faceapi
  .detectAllFaces(input)                       // SsdMobilenetv1Options by default
  .withFaceLandmarks()                         // enables alignment
  .withFaceDescriptors();                      // 128-d embeddings

const matcher = new faceapi.FaceMatcher(results, 0.6);

const displaySize = { width: input.width, height: input.height };
const canvas = faceapi.createCanvasFromMedia(input);
faceapi.matchDimensions(canvas, displaySize);
const resized = faceapi.resizeResults(results, displaySize);
faceapi.draw.drawDetections(canvas, resized);
faceapi.draw.drawFaceLandmarks(canvas, resized);
```
