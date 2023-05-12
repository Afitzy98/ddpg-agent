import * as tf from "@tensorflow/tfjs-node";

import { ExperienceReplay } from "../experience-replay";

describe("ExperienceReplay", () => {
  const capacity = 5;
  const experienceReplay = new ExperienceReplay(capacity);

  it("stores experiences", () => {
    const state = tf.tensor([0]);
    const action = tf.tensor([1]);
    const reward = 1;
    const nextState = tf.tensor([2]);
    const done = false;

    experienceReplay.add(state, action, reward, nextState, done);

    expect(experienceReplay.size()).toEqual(1);
  });

  it("limits capacity", () => {
    for (let i = 0; i < 10; i++) {
      const state = tf.tensor([i]);
      const action = tf.tensor([i + 1]);
      const reward = i;
      const nextState = tf.tensor([i + 2]);
      const done = false;

      experienceReplay.add(state, action, reward, nextState, done);
    }

    expect(experienceReplay.size()).toEqual(capacity);
  });

  it("samples experiences", () => {
    const batchSize = 3;
    const samples = experienceReplay.sample(batchSize);

    expect(samples.length).toEqual(batchSize);
  });

  it("samples different experiences", () => {
    const batchSize = 3;
    const samples1 = experienceReplay.sample(batchSize);
    const samples2 = experienceReplay.sample(batchSize);

    expect(samples1).not.toEqual(samples2);
  });
});
