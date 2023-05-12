import * as tf from "@tensorflow/tfjs-node";

export class Actor {
  private model: tf.LayersModel;

  constructor(
    inputShape: number[],
    outputShape: number[],
    learningRate: number,
    hiddenUnits: number[]
  ) {
    this.model = this.createModel(inputShape, outputShape, hiddenUnits);
    this.compileModel(learningRate);
  }

  private createModel(
    inputShape: number[],
    outputShape: number[],
    hiddenUnits: number[]
  ): tf.LayersModel {
    const input = tf.input({ shape: inputShape });

    let previousLayer = input;
    hiddenUnits.forEach((units) => {
      previousLayer = tf.layers
        .dense({ units, activation: "relu" })
        .apply(previousLayer) as tf.SymbolicTensor;
    });

    const output = tf.layers
      .dense({ units: outputShape[0], activation: "tanh" })
      .apply(previousLayer) as tf.SymbolicTensor;

    return tf.model({ inputs: input, outputs: output });
  }

  private compileModel(learningRate: number): void {
    this.model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: "meanSquaredError",
    });
  }

  public getModel(): tf.LayersModel {
    return this.model;
  }

  public predict(input: tf.Tensor): tf.Tensor {
    return tf.tidy(() => this.model.predict(input) as tf.Tensor);
  }
}

export class Critic {
  private model: tf.LayersModel;

  constructor(
    stateShape: number[],
    actionShape: number[],
    learningRate: number,
    hiddenUnits: number[]
  ) {
    this.model = this.createModel(stateShape, actionShape, hiddenUnits);
    this.compileModel(learningRate);
  }

  private createModel(
    stateShape: number[],
    actionShape: number[],
    hiddenUnits: number[]
  ): tf.LayersModel {
    const stateInput = tf.input({ shape: stateShape });
    const actionInput = tf.input({ shape: actionShape });
    const concatenated = tf.layers
      .concatenate()
      .apply([stateInput, actionInput]) as tf.SymbolicTensor;

    let previousLayer = concatenated;
    hiddenUnits.forEach((units) => {
      previousLayer = tf.layers
        .dense({ units, activation: "relu" })
        .apply(previousLayer) as tf.SymbolicTensor;
    });

    const output = tf.layers
      .dense({ units: 1 })
      .apply(previousLayer) as tf.SymbolicTensor;

    return tf.model({ inputs: [stateInput, actionInput], outputs: output });
  }

  private compileModel(learningRate: number): void {
    this.model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: "meanSquaredError",
    });
  }

  public getModel(): tf.LayersModel {
    return this.model;
  }

  public predict(state: tf.Tensor, action: tf.Tensor): tf.Tensor {
    return tf.tidy(() => this.model.predict([state, action]) as tf.Tensor);
  }
}
