import * as tf from "@tensorflow/tfjs-node";

export class OrnsteinUhlenbeckProcess {
  private xPrev: tf.Tensor | null;

  constructor(
    private mu: tf.Tensor,
    private sigma: number,
    private theta: number = 0.15,
    private dt: number = 1e-2
  ) {
    this.xPrev = null;
  }

  public sample(): tf.Tensor {
    const xPrev = this.xPrev || this.mu;
    const noiseSample = tf.randomNormal(xPrev.shape, 0, this.sigma);
    const x = xPrev
      .add(this.mu.sub(xPrev).mul(this.theta * this.dt))
      .add(noiseSample.mul(tf.scalar(Math.sqrt(this.dt))));
    this.xPrev = x;
    return x;
  }

  public reset(): void {
    this.xPrev = null;
  }
}
