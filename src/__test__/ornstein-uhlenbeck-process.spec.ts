import * as tf from "@tensorflow/tfjs-node";

import { OrnsteinUhlenbeckProcess } from "../ornstein-uhlenbeck-process";

describe("OrnsteinUhlenbeckProcess", () => {
  const mu = tf.scalar(0);
  const sigma = 0.2;
  const theta = 0.15;
  const dt = 1e-2;

  it("samples from the process", () => {
    const ouProcess = new OrnsteinUhlenbeckProcess(mu, sigma, theta, dt);
    const sample = ouProcess.sample() as tf.Tensor1D;

    const minValue = mu.sub(tf.scalar(sigma * 3)).dataSync()[0];
    const maxValue = mu.add(tf.scalar(sigma * 3)).dataSync()[0];

    expect(sample.dataSync()[0]).toBeGreaterThanOrEqual(minValue);
    expect(sample.dataSync()[0]).toBeLessThanOrEqual(maxValue);
  });

  it("samples are different", () => {
    const ouProcess = new OrnsteinUhlenbeckProcess(mu, sigma, theta, dt);
    const sample1 = ouProcess.sample() as tf.Tensor1D;
    const sample2 = ouProcess.sample() as tf.Tensor1D;

    expect(sample1.dataSync()[0]).not.toEqual(sample2.dataSync()[0]);
  });

  it("resets the process, using default params", () => {
    const ouProcess = new OrnsteinUhlenbeckProcess(mu, sigma);
    ouProcess.sample();
    ouProcess.reset();

    const sample = ouProcess.sample() as tf.Tensor1D;

    const minValue = mu.sub(tf.scalar(sigma * 3)).dataSync()[0];
    const maxValue = mu.add(tf.scalar(sigma * 3)).dataSync()[0];

    expect(sample.dataSync()[0]).toBeGreaterThanOrEqual(minValue);
    expect(sample.dataSync()[0]).toBeLessThanOrEqual(maxValue);
  });
});
