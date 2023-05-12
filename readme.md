<h1 align="center" style="border-bottom: none;">🧠 DDPG Agent</h1>
<h3 align="center">A JavaScript library implementing the Deep Deterministic Policy Gradient (DDPG) algorithm with TensorFlow.js</h3>
<p align="center">
  <a href="https://www.npmjs.com/package/ddpg-agent">
    <img alt="npm latest version" src="https://img.shields.io/npm/v/ddpg-agent/latest.svg">
  </a>
  <a href="https://github.com/Afitzy98/ddpg-agent/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/Afitzy98/ddpg-agent">
  </a>
  <a href="https://github.com/Afitzy98/ddpg-agent/blob/main/LICENSE">
    <img alt="License" src="https://img.shields.io/github/license/Afitzy98/ddpg-agent">
  </a>
  <a href="#badge">
    <img alt="Co-authored by GPT-4" src="https://img.shields.io/badge/Co--authored%20by-GPT--4-blue">
  </a>
</p>

DDPG Agent is a JavaScript library for implementing reinforcement learning agents using the Deep Deterministic Policy Gradient (DDPG) algorithm. It leverages TensorFlow.js for neural network computations and supports a continuous action space, making it ideal for complex tasks such as robotic control and other high-dimensionality problems.

## Features

- Actor-Critic architecture with target networks.
- Ornstein-Uhlenbeck process for exploration noise.
- Experience Replay for stabilizing learning.
- Supports custom configuration of agent parameters.
- Save and load models for continued training or deployment.
- Compatible with TensorFlow.js Node backend for GPU-accelerated computations.

## Installation

Install the package using npm:

```bash
npm install ddpg-agent
```

or yarn:

```bash
yarn add ddpg-agent
```

## Usage

```javascript
import * as tf from "@tensorflow/tfjs-node";
import { DDPGAgent } from "ddpg-agent";

// Define your agent configuration
const agentConfig = {
  stateShape: [8],
  actionShape: [2],
  actorLearningRate: 0.001,
  criticLearningRate: 0.002,
  bufferSize: 10000,
  batchSize: 64,
  gamma: 0.99,
  tau: 0.005,
  noiseMu: 0,
  noiseSigma: 0.2,
  actorHiddenUnits: [400, 300],
  criticHiddenUnits: [400, 300],
};

// Initialize the agent
const agent = new DDPGAgent(agentConfig);

// In your training loop:
// 1. Call agent.act(state) to get action based on current state.
// 2. Execute action in environment to get next state and reward.
// 3. Store experience in agent's replay buffer.
// 4. Call agent.train() to update actor and critic networks.
```

## API

### DDPGAgent

The main class for creating a Deep Deterministic Policy Gradient (DDPG) agent.

`constructor(params: DDPGAgentConfig)`
Create a new instance of the DDPGAgent.

params: A DDPGAgentConfig object with the parameters for the DDPGAgent.
`act(state: tf.Tensor): tf.Tensor`
Generates an action for the given state.

state: The current state for which to generate an action.
`train(): Promise<{ actorLoss: number; criticLoss: number }>`
Performs a training step of the DDPGAgent. Returns a promise that resolves to the actor and critic loss values.

`saveModels(): void`
Saves the actor and critic models of the DDPGAgent to disk.

`loadModels(): void`
Loads the actor and critic models of the DDPGAgent from disk.

## Implementation details

The DDPGAgent uses the Actor-Critic method, specifically the Deep Deterministic Policy Gradient (DDPG) algorithm. This method uses two networks, an actor and a critic, that work together to learn the optimal policy. The actor network takes the current state as input and outputs the action to take, while the critic network takes the current state and the actor's output action as input, and outputs a value function estimating the future rewards.

To handle the exploration-exploitation trade-off, the DDPGAgent uses an Ornstein-Uhlenbeck process to add noise to the actor's output actions. This encourages the agent to explore the environment.

The DDPGAgent also includes a replay buffer, which stores the agent's experiences and samples from them in a random order to remove the correlation in the observation sequence and prevent action values from oscillating or diverging catastrophically.

## Author

- Aaron Fitzpatrick ([Afitzy98](https://github.com/Afitzy98))

## Contributing

Contributions are welcome! Please open an issue or submit a pull request on the GitHub repository.
