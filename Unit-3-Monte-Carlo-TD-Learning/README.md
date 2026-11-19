# Unit III – Monte Carlo Methods and Temporal-Difference Learning

## 1. Model-free prediction and control
- **Prediction:** estimate vπ(s) or qπ(s,a) for a given policy.
- **Control:** find/improve a policy to maximize expected return.
Model-free methods learn from sampled experience without requiring the transition/reward model.

## 2. Monte Carlo (MC) methods
MC estimates values by averaging returns observed in episodes. It does not bootstrap and ordinarily updates after an episode terminates.

```math
V(s) = average of observed returns following visits to s
V(s) ← V(s)+α[G_t−V(s)]
```
- **First-visit MC:** use only the first visit to a state in each episode.
- **Every-visit MC:** use every visit to the state in each episode.

### MC policy evaluation steps
1. Generate an episode following π.
2. For each visited state, calculate the return G from that time.
3. Apply first-visit or every-visit update.
4. Repeat over episodes.

## 3. On-policy and off-policy
- On-policy: learn/evaluate the policy used to generate behaviour (e.g. SARSA).
- Off-policy: learn a target policy using data from a behaviour policy (e.g. Q-learning).
- Behaviour policy μ generates actions/data.
- Target policy π is the policy being evaluated or learned.

## 4. Importance sampling
For a single action:
```math
ρ_t = π(A_t|S_t) / μ(A_t|S_t)
```
For a trajectory segment:
```math
ρ_{t:T−1}=Π_{k=t}^{T−1} [π(A_k|S_k)/μ(A_k|S_k)]
```
It reweights behaviour-policy samples to estimate target-policy expectations. Every action selected by the target policy must have support under the behaviour policy for ordinary importance sampling to work.

## 5. Temporal Difference learning
TD learns from each transition and bootstraps from the estimated next-state value.

TD target:
```math
R_{t+1}+γV(S_{t+1})
```
TD error:
```math
δ_t=R_{t+1}+γV(S_{t+1})−V(S_t)
```
TD(0) update:
```math
V(S_t)←V(S_t)+αδ_t
```
For terminal next states, use V(terminal)=0. It is called TD(0) because it uses a one-step target.

## 6. DP vs MC vs TD
| Property | DP | MC | TD |
|---|---|---|---|
| Model required | Yes | No | No |
| Target | Expected backup | Complete sampled return | One-step reward + estimated next value |
| Bootstrapping | Yes | No | Yes |
| Update | Model-based sweeps | After episode | Each step |
| Sample experience | Not required if model known | Required | Required |

MC target is the actual return; TD target bootstraps; DP target is an expectation over the model.

## 7. Batch updating
Collect a fixed set of episodes/transitions and repeatedly update estimates using that dataset until the batch solution is approached. Reusing data can reduce sampling noise but does not make the data independent.

## 8. Exploring starts
Assumption that every state-action pair has a non-zero probability of being selected as the initial pair. It ensures all pairs can be explored in theoretical MC control. In practice, exploration can instead be encouraged by stochastic/ε-soft policies.

## 9. SARSA (on-policy TD control)
SARSA name: State, Action, Reward, next State, next Action.
```math
Q(S_t,A_t)←Q(S_t,A_t)+α[
R_{t+1}+γQ(S_{t+1},A_{t+1})−Q(S_t,A_t)]
```
Steps:
1. Initialize Q; choose action using ε-greedy policy.
2. Take action; observe reward and next state.
3. Choose next action using the same behaviour policy.
4. Update Q using the SARSA equation.
5. Move to next state/action and repeat.

## 10. Q-Learning (off-policy TD control)
```math
Q(S_t,A_t)←Q(S_t,A_t)+α[
R_{t+1}+γ max_a Q(S_{t+1},a)−Q(S_t,A_t)]
```
It uses the greedy next-state action value in its target, even if behaviour explores.

## 11. Expected SARSA
```math
Q(S_t,A_t)←Q(S_t,A_t)+α[
R_{t+1}+γΣ_a π(a|S_{t+1})Q(S_{t+1},a)−Q(S_t,A_t)]
```
It uses the expected next action value under π rather than one sampled next action or a maximum.

## 12. SARSA vs Q-learning and cliff walking
| SARSA | Q-learning |
|---|---|
| On-policy | Off-policy |
| Target uses Q(S',A') | Target uses max_a Q(S',a) |
| Learns value of behaviour policy | Learns greedy target values |
| Accounts for exploratory next action | Target ignores exploratory next action |

In the cliff-walking example, exploratory actions near the cliff can be dangerous. SARSA accounts for this risk and can learn a safer route; Q-learning's greedy target favours the shortest optimal route, while behaviour may still explore.

## 13. Maximization bias and Double Q-learning
Maximizing over noisy action-value estimates tends to select positive estimation errors, causing overestimation. Double Q-learning maintains two estimators and separates action selection from action evaluation:
```math
Q_1(s,a)←Q_1(s,a)+α[r+γQ_2(s',argmax_b Q_1(s',b))−Q_1(s,a)]
```
The update for Q2 is symmetric. This reduces maximization bias.

## 14. Rejection sampling
A sampling method that accepts a proposed sample with a specified probability and rejects it otherwise. It can waste computation when acceptance probability is low.

## Quick recall
- MC: complete return, no bootstrap.
- TD: one-step target, bootstrap, online updates.
- SARSA: next action actually selected.
- Q-learning: maximum next Q.
- Expected SARSA: expectation over next actions.
