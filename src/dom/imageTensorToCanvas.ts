import * as tf from '@tensorflow/tfjs-core';

import { env } from '../env';
import { isTensor4D } from '../utils';

export async function imageTensorToCanvas(
  imgTensor: tf.Tensor,
  canvas?: HTMLCanvasElement
): Promise<HTMLCanvasElement> {

  const targetCanvas = canvas || env.getEnv().createCanvasElement()

  const [height, width, numChannels] = imgTensor.shape.slice(isTensor4D(imgTensor) ? 1 : 0)
  const imgTensor3D = tf.tidy(() => tf.cast(tf.reshape<tf.Rank.R3>(imgTensor, [height, width, numChannels]), 'int32'))
  await tf.browser.toPixels(imgTensor3D, targetCanvas)

  imgTensor3D.dispose()

  return targetCanvas
}