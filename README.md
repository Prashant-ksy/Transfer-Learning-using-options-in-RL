# Transfer Learning with SR-Derived Options in Reinforcement Learning

# Overview

This project investigates **temporal abstraction** and **skill transfer** in Reinforcement Learning (RL) using:

- Successor Representation (SR)
- Eigenoptions
- Agent-space portable options

The goal is to learn **reusable high-level skills (options)** in one environment and transfer them to new environments to improve:

- Exploration efficiency  
- Jumpstart performance  
- Convergence speed  
- Diffusion across environment structures  

Two major concepts are used:

---

## 1. SR-Based Eigenoptions

The Successor Representation encodes long-term state visitation.  
By computing eigenvectors of the SR matrix, we obtain **eigenoptions**, which define intrinsic reward gradients and structured exploratory behaviors.

Eigenoptions provide:

- Task-independent navigation skills  
- Movement along structural directions of the map  
- Fast generalization when the task changes but the environment remains the same  

---

## 2. Agent-Space Portable Options

To support **cross-environment transfer**, we represent states using an **agent-centric encoding**:

```
(dy, dx) = (door_y – agent_y, door_x – agent_x)
```

This forms a relative, translation-invariant state representation.  
A policy learned in this agent-space becomes **portable** because it does not depend on:

- Absolute coordinates  
- Room size  
- Door positions  
- Environment layout  

Such options can be directly reused in new Four-Rooms environments with different wall and door configurations.

---

# Experiments

Three experiments were conducted to evaluate SR-based eigenoptions and agent-space options.

---

## Experiment 1 — Goal Change in the Same Environment

### Objective
Test whether eigenoptions generalize when only the **goal state** changes and the **environment stays identical**.

### Method
- Compute SR  
- Extract eigenoptions  
- Train Q-learning with and without options  
- Change the goal location  

### Results
- Option-augmented agent reaches **100% success almost instantly**  
- Baseline Q-learning requires many episodes  
- Eigenoptions serve as strong reusable navigation priors  

---

## Experiment 2 — Transfer to a New Four-Rooms Layout (Hybrid Agent-Space Skill)

### Objective
Test whether a portable skill learned in a **11×5 environment** can transfer to a full **Four-Rooms** environment with different geometry.

### Method
- Learn SR in *agent-space*  
- Learn a portable eigen-policy  
- Transfer the policy to a new Four-Rooms map  
- Train:
  - Baseline Q-learner  
  - High-level agent with transferred option  

### Results
- Transferred option provides a **jumpstart boost**  
- Temporary backlash occurs while learning when to use the option  
- Final performance is significantly better than baseline  

### Interpretation
Agent-space encoding enables **geometry-invariant skills**, improving transfer performance.

---

## Experiment 3 — Diffusion Time Across 20 Randomized Four-Rooms Environments

### Objective
Measure how quickly agents traverse randomized environments using **Mean First Passage Time (MFPT)** across random start-goal pairs.

### Method
For each of 20 randomly generated maps:
- Train a primitive agent  
- Train an agent with transferred options  
- Compute MFPT over 80 random pairs  

### Results
- Transferred options reduce diffusion time in **most environments**  
- Primitive agents diffuse more slowly and irregularly  
- Options help cross bottlenecks (doors) more efficiently  

