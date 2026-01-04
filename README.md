# Reinforcement Learning Golf AI

A vectorized Reinforcement Learning environment where an AI agent learns to play golf using Proximal Policy Optimization (PPO).

## Project Overview

This project simulates a physics-based 2D golf environment where an intelligent agent learns to putt a ball into a hole. Unlike traditional single-threaded simulations, this project employs a **Vectorized Physics Engine** that simulates **500 parallel processes** simultaneously. This allows the AI to gather experience 500x faster than real-time, leading to rapid convergence.

The default mode puts you in control of the setup (Ball, Hole & **Barriers**), while the AI takes over the execution (Angle & Power), learning from its mistakes in real-time.

## Theory & Architecture

### 1. Vectorized Training (Parallel Universes)

Training RL agents on physical tasks is notoriously slow because interaction with the environment is the bottleneck.

- **Traditional approach**: Agent acts -> Physics Step -> Agent acts. (Serial)
- **Vectorized approach**: The generic `VectorPhysicsEngine` maintains `N=500` independent game states.
  - The Agent observes `500` state vectors.
  - The Agent outputs `500` action vectors.
  - The Physics Engine updates 500 ball positions in a single numpy operation.
  - **Noise Injection**: During reset, slight random noise is added to the ball positions of the training environments (indices 1-499) to prevent "identical universe" overfitting and reward oscillation.

### 2. The RL Agent (PPO) & Perception

We use a custom implementation of **Proximal Policy Optimization (PPO)**.

- **State Space (10 Dimensions)**:
  - **Target Vector (2)**: Relative position `(ball_x - hole_x, ball_y - hole_y)`.
  - **LIDAR / Raycast (8)**: The agent casts 8 rays in a circle to detect distances to Walls and **User-drawn Barriers**. This gives the agent "eyes" to see obstacles and navigate around them.
- **Network**:
  - **Input**: 10 units.
  - **Hidden Layers**: Two fully connected layers with `128` units each (`Tanh` activation).
  - **Output (Actor)**: Mean `Angle` and `Power`.
  - **Output (Critic)**: Estimated Value `V(s)`.

### 3. Physics Model

The physics engine is a custom rigid-body simulator built on NumPy.

- **Friction**: A constant deceleration force.
- **Collisions**: Boundaries act as elastic walls.
- **Barriers**: Users can draw rectangular wooden barriers. The engine handles AABB collision detection and velocity reflection.
- **Hole Mechanics**: A pure distance check.

### 4. Reward Structure

- **Distance Penalty**: `-Distance * 10`.
- **Success Bonus**: `+20.0`.

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
  - **Shift + Left Drag**: Draw a Barrier.
  - **Shift + Right Click**: Remove a Barrier.
  - **Enter**: Lock in positions and start Training Mode.

- **Training Mode**:
  - **Speed Slider**: Drag the slider in the top right to accelerate training (up to 500x speed).
  - **Esc**: Return to Setup Mode to create a new scenario.

## Visuals

- **3D Ball**: Rendered with pseudo-3D shading.
- **LIDAR**: (Implicit) The agent reacts to obstacles detected by its internal sensors.
- **HUD**: Real-time stats including Success Rate (moving average).
