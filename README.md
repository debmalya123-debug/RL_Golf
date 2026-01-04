# Reinforcement Learning Golf AI

A vectorized Reinforcement Learning environment where an AI agent learns to play golf using Proximal Policy Optimization (PPO).

## Project Overview

This project simulates a physics-based 2D golf environment where an intelligent agent learns to putt a ball into a hole. Unlike traditional single-threaded simulations, this project employs a **Vectorized Physics Engine** that simulates **100 parallel processes** simultaneously. This allows the AI to gather experience 100x faster than real-time, leading to rapid convergence.

The default mode puts you in control of the setup (Ball & Hole positions), while the AI takes over the execution (Angle & Power), learning from its mistakes in real-time.

## Theory & Architecture

### 1. Vectorized Training (Parallel Universes)

Training RL agents on physical tasks is notoriously slow because interaction with the environment is the bottleneck.

- **Traditional approach**: Agent acts -> Physics Step -> Agent acts. (Serial)
- **Vectorized approach**: The generic `VectorPhysicsEngine` maintains `N=100` independent game states.
  - The Agent observes `(100, 2)` state vectors.
  - The Agent outputs `(100, 2)` action vectors.
  - The Physics Engine updates 100 ball positions in a single numpy operation.
  - This massively improves "Sample Efficiency per Second" without needing multi-threading or multi-processing overhead.

### 2. The RL Agent (PPO)

We use a custom implementation of **Proximal Policy Optimization (PPO)**, a state-of-the-art on-policy gradient method.

- **Algorithm**: PPO-Clip
- **Actor-Critic Network**:
  - **Input**: Relative position vector `(ball_x - hole_x, ball_y - hole_y)`.
  - **Hidden Layers**: Two fully connected layers with `128` units each.
  - **Output (Actor)**: Mean `Angle` (radians) and `Power`. The agent samples from a Normal distribution around these means to explore.
  - **Output (Critic)**: Estimated Value `V(s)` of the current state.
- **Optimization Details**:
  - **Orthogonal Initialization**: Weights are initialized orthogonally to preserve gradient flow during early training.
  - **Gradient Clipping**: Prevents exploding gradients.
  - **Reward Normalization**: Rewards are normalized batch-wise to keep the optimization landscape stable.

### 3. Physics Model

The physics engine is a custom rigid-body simulator built on NumPy.

- **Friction**: A constant deceleration force is applied opposite to velocity.
- **Collisions**: Boundaries act as elastic walls with a restitution coefficient of 0.7 (balls lose 30% energy on bounce).
- **Hole Mechanics**: A pure distance check. If `dist(ball, hole) < radius`, the ball is "in".

### 4. Reward Structure

The agent learns purely from a sparse/dense hybrid reward signal:

- **Distance Penalty**: `-Distance * 10`. The closer the ball stops to the hole, the less negative penalty it receives.
- **Success Bonus**: `+20.0`. A massive spike in reward for successfully holing the putt.

## Installation

This project requires Python 3.8+.

1.  Clone the repository.
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
    - `numpy`: core vector math.
    - `pygame`: real-time visualization.
    - `torch`: deep learning framework.

## How to Run

Execute the main script:

```bash
python main.py
```

### Controls

- **Setup Mode** (Default):
  - **Left Click**: Place the Ball.
  - **Right Click**: Place the Hole.
  - **Enter**: Lock in positions and start Training Mode.
- **Training Mode**:
  - **Speed Slider**: Drag the slider in the top right to accelerate training (up to 500x speed).
  - **Esc**: Return to Setup Mode to create a new scenario.
