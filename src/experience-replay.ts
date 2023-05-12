import * as tf from "@tensorflow/tfjs-node";

interface Experience {
  state: tf.Tensor;
  action: tf.Tensor;
  reward: number;
  nextState: tf.Tensor;
  done: boolean;
}

export class ExperienceReplay {
  private buffer: Experience[];
  private bufferSize: number;
  private index: number;

  constructor(bufferSize: number) {
    this.buffer = [];
    this.bufferSize = bufferSize;
    this.index = 0;
  }

  public add(
    state: tf.Tensor,
    action: tf.Tensor,
    reward: number,
    nextState: tf.Tensor,
    done: boolean
  ): void {
    const experience: Experience = {
      state,
      action,
      reward,
      nextState,
      done,
    };

    if (this.buffer.length < this.bufferSize) {
      this.buffer.push(experience);
    } else {
      this.buffer[this.index] = experience;
      this.index = (this.index + 1) % this.bufferSize;
    }
  }

  public sample(batchSize: number): Experience[] {
    if (this.buffer.length < batchSize) {
      throw new Error(
        `Sample size (${batchSize}) is greater than buffer size (${this.buffer.length}).`
      );
    }
    const indices = Array.from(
      { length: this.buffer.length },
      (_, i) => i
    ).slice(0, batchSize);

    tf.util.shuffle(indices);

    return indices.map((index) => this.buffer[index]);
  }

  public size(): number {
    return this.buffer.length;
  }
}
