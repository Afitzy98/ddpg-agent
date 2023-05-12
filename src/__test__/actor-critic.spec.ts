import * as tf from "@tensorflow/tfjs-node";

import { Actor, Critic } from "../actor-critic";

describe("Actor", () => {
  const inputShape = [4];
  const outputShape = [2];
  const learningRate = 0.001;
  const hiddenUnits = [32, 32];

  it("creates a model with the correct input and output shapes", () => {
    const actor = new Actor(inputShape, outputShape, learningRate, hiddenUnits);
    const model = actor.getModel();

    expect(model.inputs[0].shape.slice(1)).toEqual(inputShape);
    expect(model.outputs[0].shape.slice(1)).toEqual(outputShape);
  });

  it("predicts actions given a state", () => {
    const actor = new Actor(inputShape, outputShape, learningRate, hiddenUnits);
    const state = tf.tensor2d([[0, 1, 2, 3]]);
    const action = actor.predict(state);

    expect(action.shape).toEqual([1].concat(outputShape));
  });
});

describe("Critic", () => {
  const stateShape = [4];
  const actionShape = [2];
  const learningRate = 0.001;
  const hiddenUnits = [32, 32];

  it("creates a model with the correct input and output shapes", () => {
    const critic = new Critic(
      stateShape,
      actionShape,
      learningRate,
      hiddenUnits
    );
    const model = critic.getModel();

    expect(model.inputs[0].shape.slice(1)).toEqual(stateShape);
    expect(model.inputs[1].shape.slice(1)).toEqual(actionShape);
    expect(model.outputs[0].shape.slice(1)).toEqual([1]);
  });

  it("predicts state-action value given a state and action", () => {
    const critic = new Critic(
      stateShape,
      actionShape,
      learningRate,
      hiddenUnits
    );
    const state = tf.tensor2d([[0, 1, 2, 3]]);
    const action = tf.tensor2d([[0.5, -0.5]]);
    const value = critic.predict(state, action);

    expect(value.shape).toEqual([1, 1]);
  });
});
