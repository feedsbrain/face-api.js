# Migration: Node 22 + TensorFlow.js 4.x

Branch: `modernize/node22-tfjs4`

This document records the modernization of face-api.js from its 2020-era
toolchain (`@tensorflow/tfjs-core@1.7.0`, TypeScript 3.8, Rollup 2, Node ≤12) to
a current one (`@tensorflow/tfjs-core@4.22`, TypeScript 5.9, Rollup 4, Node
18/20/22).

**The model code and shipped weights are unchanged.** Every edit is either a
dependency bump, a build-config change, or a mechanical rewrite of a removed
TensorFlow.js API to its supported equivalent. Numerical output is verified
against the repository's own frozen fixtures — see [Validation](#validation).

---

## Status

| Area | State |
|---|---|
| Dependencies install on Node 22 (`canvas`, `tfjs-node` native addons) | ✅ done |
| `src/` compiles under TypeScript 5.9 + tfjs-core 4.22 | ✅ done (`npm run tsc`, `npm run tsc-es6`) |
| UMD bundle (`dist/face-api.js`, `.min.js`) builds under Rollup 4 | ✅ done (`npm run rollup`, `npm run rollup-min`) |
| Full build pipeline | ✅ done (`npm run build`) |
| Runtime + numeric correctness on Node | ✅ verified via `npm run smoke` (see below) |
| GitHub Actions CI (Node 18/20/22) | ✅ added (`.github/workflows/ci.yml`), replaces Travis |
| Full Karma/Jasmine test suite port | ⏳ **pending** — see [Test suite](#test-suite-pending) |
| Browser test runner | ⏳ pending (Karma removed; `@web/test-runner` or Playwright recommended) |
| `package.json` `exports` map / stop committing `dist/` | ⏳ optional follow-up |
| `@tensorflow/tfjs-core` as `peerDependency` | ⏳ optional follow-up (kept as a regular `dependency` for now) |

---

## Dependency changes

| Package | Before | After | Notes |
|---|---|---|---|
| `@tensorflow/tfjs-core` | `1.7.0` | `^4.22.0` | still a regular `dependency`; re-exported as `faceapi.tf` |
| `@tensorflow/tfjs-node` | `1.7.0` (dev) | `^4.22.0` (dev) | native addon; prebuilt binary resolves on Node 22 |
| `canvas` | `2.6.1` (dev) | `^3.1.0` (dev) | 2.6.1 does not build on Node ≥18 |
| `tslib` | `^1.11.1` | `^2.8.1` | |
| `typescript` | `^3.8.3` (dev) | `~5.9.3` (dev) | |
| `@types/node` | `^13` (dev) | `^22` (dev) | |
| `rollup` | `^2.1.0` (dev) | `^4.27.0` (dev) | |
| `rollup-plugin-typescript2` | `^0.26` | removed | → `@rollup/plugin-typescript@^12` |
| `rollup-plugin-node-resolve` | `^5.2` | removed | → `@rollup/plugin-node-resolve@^15` |
| `rollup-plugin-commonjs` | `^10.1` | removed | → `@rollup/plugin-commonjs@^28` |
| `rollup-plugin-uglify` | `^6.0` | removed | → `@rollup/plugin-terser@^0.4` |
| `typedoc` | (implicit) | `^0.28.20` (dev) | needed for TS 5.9 peer range |
| `karma*`, `jasmine*`, `ts-node` | dev | **removed** | test harness not yet ported (see below) |

`package-lock.json` was regenerated from scratch (the old tree was
unresolvable).

---

## Source changes

All changes are in response to TypeScript compile errors from the newer
tfjs-core type definitions. No algorithm, constant, layer, or weight path was
touched.

### 1. Removed chained tensor methods → functional ops

In tfjs 3.x+, `@tensorflow/tfjs-core` no longer augments `Tensor.prototype` with
chained arithmetic/shape ops unless the chained-ops registration module is
imported. Rather than take a deep-import dependency on
`dist/public/chained_ops/register_all_chained_ops`, the affected call sites were
rewritten to the functional form that the rest of the codebase already uses.

| Old (chained) | New (functional) | Files |
|---|---|---|
| `x.toFloat()` | `tf.cast(x, 'float32')` | `NetInput`, `FaceRecognitionNet`, `SsdMobilenetv1`, `TinyYolov2Base`, `Mtcnn`, `extractImagePatches`, `padToSquare` |
| `x.toInt()` | `tf.cast(x, 'int32')` | `imageTensorToCanvas` |
| `x.as1D() / as2D(a,b) / as3D(...) / as4D(...)` | `tf.reshape<tf.Rank.Rn>(x, [...])` | `AgeGenderNet`, `FaceProcessor`, `FaceLandmark68NetBase`, `NetInput`, `extractFaceTensors`, `imageTensorToCanvas` |
| `x.div(y)` | `tf.div(x, y)` | `FaceFeatureExtractor`, `TinyFaceFeatureExtractor`, `TinyXception`, `FaceRecognitionNet`, `FaceLandmark68NetBase`, `TinyYolov2Base` |
| `x.mul(y)` / `x.sub(y)` | `tf.mul(x, y)` / `tf.sub(x, y)` | `FaceLandmark68NetBase` |
| `x.mean(axes)` | `tf.mean(x, axes)` | `FaceRecognitionNet` |
| `x.expandDims()` | `tf.expandDims(x)` | `NetInput`, `Mtcnn`, `TinyYolov2Base` |
| `x.reshape([...])` | `tf.reshape<tf.Rank.Rn>(x, [...])` | `TinyYolov2Base` |
| `x.slice(begin, size)` | `tf.slice(x, begin, size)` | `TinyYolov2Base` |

`as2D(a, -1)` / `as1D()` are exact reshapes, so `.as2D(rows, -1)` became
`tf.reshape<tf.Rank.R2>(x, [rows, -1])` and `.as1D()` became
`tf.reshape<tf.Rank.R1>(x, [-1])`.

### 2. `NeuralNetwork.loadFromDisk`

`readFile(path).then(buf => buf.buffer)` now needs `as ArrayBuffer` — Node's
`Buffer.buffer` is typed `ArrayBufferLike` (`ArrayBuffer | SharedArrayBuffer`)
under `@types/node@22`, and `tf.io.weightsLoaderFactory` wants `ArrayBuffer[]`.
Runtime behavior is unchanged (weight shards from `fs.readFile` are unpooled).

### 3. Stricter TypeScript (removal of `suppressImplicitAnyIndexErrors`)

That compiler option was removed in TS 5.5. The `noImplicitAny` index-access
errors it was hiding are now handled explicitly:

- `src/env/createNodejsEnv.ts` — `global['Canvas']` etc. via a single
  `const g = global as any`.
- `src/faceExpressionNet/FaceExpressions.ts` — `(this as any)[expression]` when
  populating the dynamic expression fields.
- `src/factories/WithFaceLandmarks.ts` — `(obj as any)['landmarks']` inside the
  `isWithFaceLandmarks` type guard (after `isWithFaceDetection` narrowed `obj`).
- `src/xception/extractParams.ts`, `extractParamsFromWeigthMap.ts` —
  `const middle_flow: Record<string, any> = {}`.

### 4. Misc strict-null / generic fixes

- `src/dom/awaitMediaLoaded.ts` — `new Promise<void>(...)`; the resolved `Event`
  value was never consumed by callers, so `resolve(e)` → `resolve()`.
- `src/globalApi/DetectFacesTasks.ts` — `res(... : undefined as any)` in
  `runAndExtendWithFaceDetection` (pre-existing: the single-face path can resolve
  `undefined`; behavior preserved, typing deferred to the suite-port cleanup).
- `src/faceRecognitionNet/FaceRecognitionNet.ts` — `tf.tidy<tf.Tensor2D>(...)`
  and `tf.matMul(...) as tf.Tensor2D` (tfjs 4 `matMul` infers `Tensor<Rank>`).

### 5. Build config

- `tsconfig.json` / `tsconfig.es6.json` — dropped `suppressImplicitAnyIndexErrors`
  and the obsolete `formatCodeOptions`; `target`/`lib` `es5`→`es2018`,
  `module` for the ES build `es6`→`es2015`; removed the non-existent `typings`
  type root.
- `rollup.config.js` → `rollup.config.mjs`, rewritten for the `@rollup/*`
  plugins.
- `.travis.yml`, `karma.conf.js`, `jasmine-node.js` removed (referenced
  uninstalled packages). CI is now `.github/workflows/ci.yml`.

---

## Validation

`npm run smoke` (`scripts/smoke.js`) runs on `@tensorflow/tfjs-node` and checks
the migrated build against the repository's committed fixtures:

```
models loaded, backend = tensorflow

  PASS  faceRecognitionNet descriptor matches fixture  — len=128 euclid=0.03566 (tol 0.1)
  PASS  tinyFaceDetector finds the 6 reference faces  — count=6 maxBoxDelta=2.0px (tol 6)
  PASS  full SSD pipeline (landmarks + expr + age/gender + descriptors)  — 6 faces

SMOKE OK
```

- **Face recognition**: the 128-D descriptor for `test/images/face1.png` is
  euclidean distance **0.036** from `test/data/faceDescriptor1.json` — well
  inside the original suite's `0.1` tolerance. The recognition backbone is
  numerically intact.
- **Detection + landmarks**: `tinyFaceDetector` localises all six reference
  faces in `test/images/faces.jpg` within **2 px** of
  `test/expectedTinyFaceDetectorBoxes.ts` (original suite tolerance ~6 px).
- **Full pipeline**: `detectAllFaces().withFaceLandmarks().withFaceExpressions()
  .withAgeAndGender().withFaceDescriptors()` returns 68 landmark points, a
  128-entry descriptor, a normalised expression distribution, and a finite
  age / valid gender for every face.

Small numeric drift is expected — tfjs-node 4.x uses oneDNN and different kernel
implementations than tfjs-core 1.7 (it prints a note to that effect on startup).
Observed drift is far below the fixture tolerances.

---

## Test suite (pending)

The original suite (`test/tests/**/*.test.ts`) is written against a
Karma + `karma-typescript` browser harness with a Node bridge
(`test/env.node.ts`, `jasmine-node.js`). `karma-typescript` is abandoned and
pulls an incompatible TypeScript peer, so the runner was removed rather than
pinned.

Recommended port:

1. **Node suite** → `vitest`. The specs mostly need `describeWithNets` /
   `getTestEnv` (`test/utils.ts`, `test/env.ts`) rewired to a vitest
   `globalSetup` that does the `monkeyPatch` + `loadFromDisk` currently in
   `test/env.node.ts`. `describeWithBackend` can drop its WebGL branch on Node.
2. **Browser suite** → `@web/test-runner` (esbuild plugin) or Playwright
   component testing, loading `@tensorflow/tfjs-backend-webgl` +
   `@tensorflow/tfjs-backend-cpu` explicitly.
3. `test/tests-legacy/` (uncompressed-weight variants) can stay skipped as they
   were under `EXCLUDE_UNCOMPRESSED=true`.

Until then, `npm test` runs `npm run build && npm run smoke`, and CI runs the
same on Node 18/20/22.

---

## Follow-ups not done here

- Add a `package.json` `"exports"` map; move `dist/` out of git and publish it as
  a release artifact.
- Consider `@tensorflow/tfjs-core` as a `peerDependency` so consumers pin one
  backend/version (mixed tfjs versions cause silent runtime bugs).
- Optional: migrate to the umbrella `@tensorflow/tfjs` package and offer a
  WASM/WebGPU backend path.
- Deprecate/remove `mtcnn` and `tinyYolov2` in a future major (see `SPEC.md`).
