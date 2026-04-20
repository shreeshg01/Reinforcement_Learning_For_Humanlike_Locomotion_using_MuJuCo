# 🦾 Reinforcement Learning for Human-Like Locomotion Using MuJoCo

> **Course:** Computational Motor Control
> **Team:** [Your Name] & [Partner's Name]

---

## 📌 Overview

This project explores whether a reinforcement learning (RL) agent can learn **stable and efficient walking behavior** in a physics simulator without being given any reference motion. The main goal is to train an agent in a MuJoCo-based locomotion environment and study whether the learned gait shows features that can be interpreted as **human-like locomotion**.

The project is motivated by ideas from computational motor control, biomechanics, and neuroscience. Instead of hard-coding movement rules, we use **trial-and-error learning through reinforcement learning** to allow the agent to discover locomotion behavior on its own.

---

## 🎯 Project Goal

The central question of this project is:

> *Can a reinforcement learning agent, given suitable reward signals, independently discover stable, efficient, and partially human-like walking behavior inside a physics simulator?*

To answer this, we use **MuJoCo** as the simulation engine and **Proximal Policy Optimization (PPO)** as the learning algorithm.

---

## ✅ Objectives

By the end of the project, we aim to:

- Set up and run a MuJoCo locomotion environment using Gymnasium
- Train a PPO-based RL agent from scratch
- Analyze whether the learned policy can generate stable forward locomotion
- Improve gait quality using reward shaping and training refinement
- Compare the learned walking strategy with basic principles of human locomotion
- Reflect on what the results suggest about motor learning in simulated systems

---

## 🌍 Proposed Environments

The project uses MuJoCo environments through Gymnasium. Depending on training stability and realism requirements, the following environments may be used:

| Environment | Description |
|-------------|-------------|
| `Walker2d-v4` | Simpler baseline — useful for training and debugging |
| `Humanoid-v4` | More human-like body structure — better for full-body locomotion goals |

---

## 🛠️ Core Technical Stack

| Tool | Purpose |
|------|---------|
| Python 3.10+ | Primary programming language |
| MuJoCo | Physics simulation |
| Gymnasium | Environment interface |
| Stable-Baselines3 | PPO training framework |
| NumPy / Matplotlib | Data handling and visualization |
| imageio / ffmpeg | Rollout video generation |

**Optional extensions:**
- `MyoSuite` — for more physiologically realistic control experiments

---

## 🧠 Learning Method

We use **Proximal Policy Optimization (PPO)** to train the locomotion policy.

The agent receives observations from the environment, chooses actions at every timestep, and improves its behavior by maximizing cumulative reward over many episodes. PPO is chosen because it is widely used in continuous-control problems and provides a strong baseline for locomotion tasks.

---

## 🏆 Reward Design

Reward design is one of the most important parts of this project. A poor reward can produce unnatural behaviors such as hopping, collapsing, spinning, or motion that technically improves reward without resembling walking.

The reward design may include the following components:

| Component | Purpose |
|-----------|---------|
| ✅ Forward velocity reward | Encourage movement in the target direction |
| ✅ Upright posture reward / penalty | Discourage excessive torso lean |
| ✅ Energy or action penalty | Reduce unnecessarily aggressive motion |
| ✅ Fall penalty | Penalize unstable or collapsed states |
| ⚙️ Contact / gait regularity terms | Optional — for later refinement if needed |

---

## 🔄 Project Pipeline

### Phase 1 — Baseline Setup
- Configure the MuJoCo environment
- Verify environment reset/step works
- Create working `train.py` and `test.py`
- Save checkpoints and replay videos

### Phase 2 — Baseline PPO Training
- Train a PPO agent using the default or minimally modified environment reward
- Evaluate whether the agent learns basic forward locomotion
- Inspect early failure modes such as falling or frozen legs

### Phase 3 — Reward Shaping and Training Refinement
- Improve posture, stability, and motion smoothness
- Compare different reward combinations
- Tune training settings if required

### Phase 4 — Gait Analysis
- Record rollout videos
- Measure episode reward, episode length, and qualitative gait quality
- Analyze stability, movement coordination, and efficiency proxies

### Phase 5 — Interpretation
- Compare learned behaviors to simplified principles of human locomotion
- Discuss the difference between *"locomotion"* and *"human-like locomotion"*
- Identify limitations of torque-driven RL control in comparison to biological systems

---

## 📁 Repository Structure

```
CMC_Project/
├── train.py
├── test.py
├── models/
├── logs/
├── videos/
├── README.md
└── requirements.txt
```

---

## 🧭 Current Direction

A key practical lesson from early implementation is that simpler locomotion environments are useful for pipeline validation, but not necessarily for realistic human-like appearance. For example:

- **Walker2d** is easier to train, but visually simplified
- **Humanoid** is more human-like in structure, but much harder to train well

Because of this, the project may use a simpler environment for early progress and a richer environment for final evaluation.

---

## 📦 Expected Deliverables

The final project is expected to include:

- [ ] A working PPO training pipeline in MuJoCo
- [ ] Saved checkpoints and trained model files
- [ ] Rollout videos of learned locomotion
- [ ] Analysis of training progress and gait quality
- [ ] A written discussion of whether the learned behavior can reasonably be described as human-like

---

## ⚠️ Limitations

This project does not claim to fully reproduce real human biomechanics or neural motor control. Even when the learned motion appears more human-like, it is still based on a simplified simulated body and reward-driven control policy.

Important limitations include:

- Simplified rigid-body physics
- Torque-driven rather than full muscle-driven control
- No direct modeling of human sensory delays or true biological neural processes
- Heavy dependence on reward design and training stability

---

## 📚 References

1. Lee et al. (2021), *Deep Reinforcement Learning for Modeling Human Locomotion Control in Neuromechanical Simulation*
2. Schulman et al. (2017), *Proximal Policy Optimization Algorithms*
3. Wang et al. (2022), *MyoSim: Fast and Physiologically Realistic MuJoCo Models for Musculoskeletal Motor Control*
4. Dembia et al. (2020), *OpenSim Moco: Musculoskeletal Optimal Control*
