import commonjs from '@rollup/plugin-commonjs';
import { nodeResolve } from '@rollup/plugin-node-resolve';
import typescript from '@rollup/plugin-typescript';
import terser from '@rollup/plugin-terser';

const { minify } = process.env;

export default {
  input: 'src/index.ts',
  plugins: [
    typescript({
      tsconfig: './tsconfig.json',
      compilerOptions: {
        module: 'ESNext',
        declaration: false,
        declarationMap: false,
        sourceMap: !minify,
        outDir: 'dist'
      },
      exclude: ['test/**', 'node_modules/**']
    }),
    nodeResolve(),
    commonjs({
      include: 'node_modules/**'
    })
  ].concat(minify ? terser() : []),
  output: {
    extend: true,
    file: `dist/face-api${minify ? '.min' : ''}.js`,
    format: 'umd',
    name: 'faceapi',
    globals: {
      crypto: 'crypto'
    },
    sourcemap: !minify
  },
  external: ['crypto'],
  onwarn: (warning) => {
    const ignoreWarnings = ['CIRCULAR_DEPENDENCY', 'CIRCULAR', 'THIS_IS_UNDEFINED', 'EVAL'];
    if (ignoreWarnings.some(w => w === warning.code)) {
      return;
    }
    if (warning.missing === 'alea') {
      return;
    }
    console.warn(warning.message);
  }
};
