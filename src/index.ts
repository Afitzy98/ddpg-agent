import * as tf from "@tensorflow/tfjs-node";

import { Actor, Critic } from "./actor-critic";
import { ExperienceReplay } from "./experience-replay";
import { OrnsteinUhlenbeckProcess } from "./ornstein-uhlenbeck-process";

export interface DDPGAgentConfig {
  stateShape: number[];
  actionShape: number[];
  actorLearningRate: number;
  criticLearningRate: number;
  bufferSize: number;
  batchSize: number;
  gamma: number;
  tau: number;
  noiseMu: number;
  noiseSigma: number;
  actorHiddenUnits: number[];
  criticHiddenUnits: number[];
}

export class DDPGAgent {
  public actor: Actor;
  public actorTarget: Actor;
  public critic: Critic;
  public criticTarget: Critic;
  public replayBuffer: ExperienceReplay;
  public noise: OrnsteinUhlenbeckProcess;
  private actorOptimizer: tf.Optimizer;
  private criticOptimizer: tf.Optimizer;

  constructor(private params: DDPGAgentConfig) {
    const {
      stateShape,
      actionShape,
      actorLearningRate,
      criticLearningRate,
      bufferSize,
      noiseMu,
      noiseSigma,
      actorHiddenUnits,
      criticHiddenUnits,
    } = params;

    this.actor = new Actor(
      stateShape,
      actionShape,
      actorLearningRate,
      actorHiddenUnits
    );
    this.actorTarget = new Actor(
      stateShape,
      actionShape,
      actorLearningRate,
      actorHiddenUnits
    );
    this.critic = new Critic(
      stateShape,
      actionShape,
      criticLearningRate,
      criticHiddenUnits
    );
    this.criticTarget = new Critic(
      stateShape,
      actionShape,
      criticLearningRate,
      criticHiddenUnits
    );
    this.replayBuffer = new ExperienceReplay(bufferSize);
    this.noise = new OrnsteinUhlenbeckProcess(tf.scalar(noiseMu), noiseSigma);
    this.actorOptimizer = tf.train.adam(actorLearningRate);
    this.criticOptimizer = tf.train.adam(criticLearningRate);

    this.updateTargets(1);
  }

  public act(state: tf.Tensor): tf.Tensor {
    const action = this.actor.predict(state);
    const noise = this.noise.sample();
    return action.add(noise).clipByValue(-1, 1);
  }

  public async train(): Promise<{ actorLoss: number; criticLoss: number }> {
    if (this.replayBuffer.size() < this.params.batchSize) {
      return { actorLoss: 0, criticLoss: 0 };
    }

    const experiences = this.replayBuffer.sample(this.params.batchSize);

    const states = tf
      .stack(experiences.map((e) => e.state))
      .reshape([this.params.batchSize, -1]);
    const actions = tf
      .stack(experiences.map((e) => e.action))
      .reshape([this.params.batchSize, -1]);
    const rewards = tf.tensor1d(experiences.map((e) => e.reward));
    const nextStates = tf
      .stack(experiences.map((e) => e.nextState))
      .reshape([this.params.batchSize, -1]);
    const dones = tf.tensor1d(experiences.map((e) => (e.done ? 1.0 : 0.0)));

    // Update actor & critic
    const criticLossValue = tf.tidy(() =>
      this.trainCritic(
        states,
        actions,
        rewards,
        nextStates,
        dones,
        this.params.gamma
      )
    );
    const actorLossValue = tf.tidy(() => this.trainActor(states));

    // Update target networks
    this.updateTargets(this.params.tau);

    // Dispose tensors
    states.dispose();
    actions.dispose();
    rewards.dispose();
    nextStates.dispose();
    dones.dispose();

    return {
      actorLoss: actorLossValue,
      criticLoss: criticLossValue,
    };
  }

  public saveModels(): void {
    this.actor.getModel().save(`file://models/actor`);
    this.actorTarget.getModel().save(`file://models/actor_target`);
    this.critic.getModel().save(`file://models/critic`);
    this.criticTarget.getModel().save(`file://models/critic_target`);
  }

  public loadModels(): void {
    tf.loadLayersModel(`file://models/actor/model.json`).then((model) => {
      this.actor.getModel().setWeights(model.getWeights());
    });
    tf.loadLayersModel(`file://models/actor_target/model.json`).then(
      (model) => {
        this.actorTarget.getModel().setWeights(model.getWeights());
      }
    );
    tf.loadLayersModel(`file://models/critic/model.json`).then((model) => {
      this.critic.getModel().setWeights(model.getWeights());
    });
    tf.loadLayersModel(`file://models/critic_target/model.json`).then(
      (model) => {
        this.criticTarget.getModel().setWeights(model.getWeights());
      }
    );
  }

  private trainActor(states: tf.Tensor): number {
    // Update actor
    let actorLossValue = 0;
    this.actorOptimizer.minimize(() => {
      return tf.tidy(() => {
        const predictedActions = this.actor.predict(states);
        const criticPredictions = this.critic.predict(states, predictedActions);
        const actorLoss = criticPredictions.mean().neg();
        actorLossValue = actorLoss.mean().dataSync()[0];
        return actorLoss.mean();
      });
    });
    return actorLossValue;
  }

  private trainCritic(
    states: tf.Tensor,
    actions: tf.Tensor,
    rewards: tf.Tensor,
    nextStates: tf.Tensor,
    dones: tf.Tensor,
    gamma: number
  ): number {
    let criticLossValue = 0;
    this.criticOptimizer.minimize(() => {
      return tf.tidy(() => {
        const nextActions = this.actorTarget.predict(nextStates);
        const targetQs = this.criticTarget.predict(nextStates, nextActions);
        const y = rewards
          .expandDims(1)
          .add(targetQs.mul(tf.scalar(1).sub(dones).mul(gamma).expandDims(1)));
        const criticPredictions = this.critic.predict(states, actions);
        const criticLoss = tf.losses.meanSquaredError(y, criticPredictions);
        criticLossValue = criticLoss.mean().dataSync()[0];
        return criticLoss.mean();
      });
    });
    return criticLossValue;
  }

  private updateTargets(tau: number): void {
    tf.tidy(() => {
      const actorWeights = this.actor.getModel().getWeights();
      const actorTargetWeights = this.actorTarget.getModel().getWeights();
      const criticWeights = this.critic.getModel().getWeights();
      const criticTargetWeights = this.criticTarget.getModel().getWeights();

      const newActorTargetWeights = actorWeights.map((weight, i) =>
        weight
          .mul(tf.scalar(1 - tau))
          .add(actorTargetWeights[i].mul(tf.scalar(tau)))
      );
      const newCriticTargetWeights = criticWeights.map((weight, i) =>
        weight
          .mul(tf.scalar(1 - tau))
          .add(criticTargetWeights[i].mul(tf.scalar(tau)))
      );

      this.actorTarget.getModel().setWeights(newActorTargetWeights);
      this.criticTarget.getModel().setWeights(newCriticTargetWeights);
    });
  }
}
