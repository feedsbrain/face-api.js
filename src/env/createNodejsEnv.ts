import { createFileSystem } from './createFileSystem';
import { Environment } from './types';

export function createNodejsEnv(): Environment {

  const g = global as any

  const Canvas = g['Canvas'] || g['HTMLCanvasElement']
  const Image = g['Image'] || g['HTMLImageElement']

  const createCanvasElement = function() {
    if (Canvas) {
      return new Canvas()
    }
    throw new Error('createCanvasElement - missing Canvas implementation for nodejs environment')
  }

  const createImageElement = function() {
    if (Image) {
      return new Image()
    }
    throw new Error('createImageElement - missing Image implementation for nodejs environment')
  }

  const fetch = g['fetch'] || function() {
    throw new Error('fetch - missing fetch implementation for nodejs environment')
  }

  const fileSystem = createFileSystem()

  return {
    Canvas: Canvas || class {},
    CanvasRenderingContext2D: g['CanvasRenderingContext2D'] || class {},
    Image: Image || class {},
    ImageData: g['ImageData'] || class {},
    Video: g['HTMLVideoElement'] || class {},
    createCanvasElement,
    createImageElement,
    fetch,
    ...fileSystem
  }
}