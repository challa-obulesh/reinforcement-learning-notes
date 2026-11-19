# Unit IV – Function Approximation Methods

## 1. Why function approximation?
Tabular methods store a separate value for each state or state-action pair. They become impractical with huge or continuous state spaces, high memory requirements and little generalization between similar states. Function approximation represents values using adjustable parameters.

```math
v̂(s,w) ≈ v_π(s),      q̂(s,a,w) ≈ q_π(s,a)
```

## 2. Linear function approximation
```math
v̂(s,w)=w^T x(s)=Σ_i w_i x_i(s)
```
x(s) is a feature vector and w is a weight vector.

Advantages: simple, efficient, relatively interpretable and mathematically tractable.
Limitations: performance depends on features and a linear model cannot represent every complex nonlinear relationship.

## 3. Neural-network function approximation
A neural network maps state/features through layers to predicted values or Q-values. It can represent nonlinear relationships and learn useful features, but needs more data/computation and can be harder to stabilize and interpret.

## 4. Gradient descent and mean squared error
For prediction target y and estimate v̂(s,w):
```math
J(w)=1/2 E[(y−v̂(S,w))²]
w ← w−α∇_w J(w)
```
The gradient update changes parameters in the direction that reduces the objective. α is the learning rate.

For a single sample and linear approximation:
```math
w ← w+α[y−w^T x]x
```

## 5. Gradient Monte Carlo
Use the complete return G_t as the target:
```math
w ← w+α[G_t−v̂(S_t,w)]∇_w v̂(S_t,w)
```
For linear features, ∇w v̂=x(S_t), so:
```math
w ← w+α[G_t−w^T x(S_t)]x(S_t)
```
It does not bootstrap and normally needs the episode return.

## 6. Semi-gradient TD(0)
TD target:
```math
U_t=R_{t+1}+γv̂(S_{t+1},w)
```
TD error:
```math
δ_t=R_{t+1}+γv̂(S_{t+1},w)−v̂(S_t,w)
```
Update:
```math
w←w+αδ_t∇_w v̂(S_t,w)
```
For linear approximation:
```math
w←w+αδ_t x(S_t)
```
It is called **semi-gradient** because the update differentiates the current estimate but treats the bootstrapped target as fixed, ignoring its dependence on w.

### Gradient MC vs semi-gradient TD
| Gradient MC | Semi-gradient TD |
|---|---|
| Complete return target | One-step bootstrapped target |
| No bootstrap | Bootstrap |
| Usually after episode | Can update each step |
| Actual sampled return | Reward + estimated next value |

## 7. TD fixed point
For linear TD(0), the expected update reaches a projected Bellman fixed point. With feature matrix Φ, transition matrix P, reward vector r and state weighting D, a common equation is:
```math
Φ^T D (r+γPΦw−Φw)=0
```
This is the projected Bellman equation.

## 8. Afterstates
An afterstate is the state immediately after the agent's action but before the environment's random response. It separates the effect of the agent's decision from environmental randomness and can simplify value estimation. Example: a board position after placing a tile but before a random tile appears.

## 9. Least-Squares TD (LSTD)
LSTD estimates linear value parameters by solving a system of equations from transition data, rather than using only incremental stochastic updates.
```math
w=A^{-1}b
A=Σ_t x_t(x_t−γx_{t+1})^T
b=Σ_t x_t R_{t+1}
```
In practice, numerical linear solvers are preferred to explicitly calculating A inverse. Related methods in the syllabus include LSTD, LSPI and LSTD-Q.

## 10. Deep Q-Network (DQN)
DQN uses a neural network to approximate the action-value function Q(s,a;w), particularly when states are high-dimensional (for example, images).

### Architecture / flow
```text
State (e.g. image)
      |
Pre-processing
      |
Online Q-network
      |
Q-values for actions
      |
ε-greedy action selection
      |
Environment -> reward, next state
      |
Replay buffer stores (s,a,r,s',done)
      |
Random mini-batch -> train online network
      |
Target network supplies stable target
```

Target:
```math
Y_t = r + γ(1−done) max_{a'} Q_target(s',a';w^-)
```
Loss:
```math
L(w)=E[(Y_t−Q_online(s,a;w))²]
```
The target network is updated periodically or slowly from the online network.

### Experience replay
Store transitions (s,a,r,s',done) in a replay buffer and sample random mini-batches for training.
Benefits:
1. Reduces correlation between consecutive samples.
2. Reuses past experience and improves data efficiency.
3. Provides varied mini-batches and helps stabilize learning.
4. Supports off-policy learning.

### Key DQN stabilizing ideas
- Experience replay buffer.
- Separate target network.
- Deep neural network for Q-value approximation.
- ε-greedy exploration is commonly used.

## 11. Risk-sensitive reinforcement learning
Risk-sensitive RL considers the distribution/variability of outcomes, not only expected return. A risk-sensitive agent may prefer a safer outcome over a high-mean but highly variable outcome.

Approaches include:
- Mean–variance objectives.
- Utility functions.
- Value at Risk (VaR) / Conditional Value at Risk (CVaR).
- Distributional value learning.

## Complexity intuition
- Tabular: memory grows with number of states (or state-action pairs).
- Linear approximation: memory grows with number of features.
- Neural approximation: memory grows with network parameters; computation is higher but can model complex functions.
