# Unit II – Markov Reward Processes, MDPs and Dynamic Programming

## 1. Markov chain / Markov process
A Markov process is a tuple (S,P), where S is the state set and P is the state-transition matrix. The Markov property says the next state depends on the current state, not the full history:
```math
P(S_{t+1}=s'|S_t=s,H_{t-1})=P(s'|s)
```
Each row of P sums to 1.

## 2. Markov Reward Process (MRP)
An MRP is (S,P,R,γ):
- S: states
- P(s'|s): transition probabilities
- R(s): expected reward in a state (or expected one-step reward)
- γ: discount factor

Return:
```math
G_t=R_{t+1}+γR_{t+2}+γ^2R_{t+3}+...
```
State value:
```math
v(s)=E[G_t|S_t=s]
```

### Bellman expectation equation for an MRP
```math
v(s)=R(s)+γΣ_{s'}P(s'|s)v(s')
```
Matrix form:
```math
v=R+γPv,   (I−γP)v=R
v=(I−γP)^{-1}R
```
The inverse solution is valid when the inverse exists, for example for γ<1 in a finite MRP.

## 3. Markov Decision Process (MDP)
An MDP is (S,A,P,R,γ). It extends an MRP by adding actions:
- S: states
- A: available actions
- P(s'|s,a): transition probability
- R(s,a,s') or expected immediate reward
- γ: discount factor

### MRP vs MDP
| MRP | MDP |
|---|---|
| No agent choice of action | Agent chooses actions |
| Has state transitions and rewards | Transitions/rewards depend on actions |
| Evaluates process values | Supports policy evaluation and control |

A policy π(a|s) specifies action probabilities. Fixing a policy in an MDP produces an MRP with policy-induced transitions and rewards:
```math
P_π(s'|s)=Σ_a π(a|s)P(s'|s,a)
R_π(s)=Σ_a π(a|s)Σ_{s'}P(s'|s,a)R(s,a,s')
```

## 4. Bellman expectation equations for an MDP
State value:
```math
v_π(s)=Σ_a π(a|s)Σ_{s'}P(s'|s,a)[R(s,a,s')+γv_π(s')]
```
Action value:
```math
q_π(s,a)=Σ_{s'}P(s'|s,a)[R(s,a,s')+γΣ_{a'}π(a'|s')q_π(s',a')]
```
Relationship:
```math
v_π(s)=Σ_a π(a|s)q_π(s,a)
```

## 5. Optimal value functions and Bellman optimality
```math
v_*(s)=max_π v_π(s)
q_*(s,a)=Σ_{s'}P(s'|s,a)[R(s,a,s')+γ max_{a'}q_*(s',a')]
v_*(s)=max_a Σ_{s'}P(s'|s,a)[R(s,a,s')+γv_*(s')]
```
An optimal deterministic policy can be obtained by:
```math
π_*(s)=argmax_a q_*(s,a)
```

## 6. Dynamic Programming (DP)
DP solves MDPs when the model (transition probabilities and rewards) is known. Its two key requirements are:
1. Optimal substructure / principle of optimality.
2. A model that provides transition probabilities and expected rewards.

### Principle of optimality
An optimal policy has the property that, after any first action and resulting state, the remaining decisions must themselves be optimal for the resulting state.

### Policy evaluation
For a fixed policy, repeatedly apply the Bellman expectation backup:
```math
V_{k+1}(s)=Σ_aπ(a|s)Σ_{s'}P(s'|s,a)[R+γV_k(s')]
```
Stop when max_s |V_(k+1)(s)−V_k(s)| < θ.

### Policy improvement
Improve the policy greedily with respect to the current value:
```math
π'(s)=argmax_a Σ_{s'}P(s'|s,a)[R(s,a,s')+γV(s')]
```

### Policy iteration algorithm
1. Initialize an arbitrary policy π.
2. **Policy evaluation:** calculate Vπ.
3. **Policy improvement:** make the policy greedy with respect to Vπ.
4. If the policy is unchanged, stop; it is optimal. Otherwise repeat steps 2–3.

```text
Initialize policy π
       |
Policy evaluation (find Vπ)
       |
Policy improvement (make π greedy)
       |
Policy stable? ---- No ----> repeat
       |
      Yes
       |
   Optimal policy
```

### Value iteration
Combine evaluation and improvement into one optimality backup:
```math
V_{k+1}(s)=max_a Σ_{s'}P(s'|s,a)[R+γV_k(s')]
```
Steps: initialize V; apply the update to every state repeatedly; stop when changes are small; extract a greedy policy. It is often more efficient than policy iteration when full policy evaluation is expensive, although convergence speed depends on the problem.

## 7. Bellman operator and contraction
For a fixed policy:
```math
(Lv)(s)=R_π(s)+γΣ_{s'}P_π(s'|s)v(s')
```
In max norm:
```math
||Lv−Lu||_∞ ≤ γ||v−u||_∞
```
For 0≤γ<1, L is a contraction. Repeated application converges to its unique fixed point vπ.

The Bellman optimality operator is also a γ-contraction in max norm for discounted finite MDPs.

## 8. Banach Fixed Point Theorem
In a complete metric space, a contraction mapping has a unique fixed point, and repeated application from any starting point converges to it. In discounted DP, the Bellman operator is a contraction, so iterative policy evaluation converges to vπ; the optimality operator converges to v*.

## 9. Norm
A norm measures the size/length of a vector and satisfies non-negativity, definiteness, homogeneity and triangle inequality. The max norm is ||v||∞=max_s |v(s)|. It is useful for bounding value-function errors and proving contraction.

## 10. Small Gridworld
A gridworld is a small state space arranged as cells. The agent moves between cells, receives rewards (often a step cost), and may encounter terminal states or boundaries. It illustrates state transitions, policies, Bellman backups, policy evaluation/improvement and value iteration.

## 11. Asynchronous DP and curse of dimensionality
- Asynchronous DP updates selected states in place rather than requiring a full synchronous sweep.
- It can focus computation on important states and use memory/time flexibly.
- Curse of dimensionality: state/action counts grow rapidly as dimensions or resolution increase, making tabular DP expensive.

## Exam comparison: Policy iteration vs value iteration
| Policy iteration | Value iteration |
|---|---|
| Alternates evaluation and improvement | Uses optimality backup directly |
| Evaluation may require many sweeps or exact solve | Partial evaluation is implicit |
| Often fewer outer iterations | Often simpler updates |
| Stops when policy is stable | Stops when value changes are small |

