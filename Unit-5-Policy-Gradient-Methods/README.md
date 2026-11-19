# Unit V – Policy Gradient Methods

## 1. Policy gradient methods
Policy gradient methods directly parameterize and optimize a policy πθ(a|s), rather than first learning a value table and acting greedily.

Objective:
```math
J(θ)=E_{π_θ}[G_t]
```
Gradient-ascent update:
```math
θ←θ+α∇_θJ(θ)
```
The aim is to maximize expected return, so gradient ascent is used.

## 2. Log-derivative trick
```math
∇_θ π_θ(a|s)=π_θ(a|s)∇_θ log π_θ(a|s)
```
Equivalently, ∇ log π = (∇π)/π when π>0. It converts derivatives of probabilities into log-probability gradients and is used to derive policy-gradient estimators.

## 3. REINFORCE
REINFORCE is a Monte Carlo policy-gradient algorithm. It samples complete episodes and updates the policy using returns:
```math
θ←θ+αG_t∇_θ log π_θ(A_t|S_t)
```
Steps:
1. Initialize policy parameters θ.
2. Generate an episode using πθ.
3. Calculate return Gt at each time step.
4. Update θ using the REINFORCE equation.
5. Repeat for many episodes.

## 4. Reward-to-go and naive REINFORCE
- Naive estimator can use the same full episode return G0 for each action.
- Reward-to-go uses only rewards from the current time onward:
```math
G_t=R_{t+1}+γR_{t+2}+γ²R_{t+3}+...
```
Reward-to-go removes past rewards that cannot be affected by the current action and often reduces variance without changing the expected gradient.

## 5. Expected Gradient Log-Probability (EGLP) lemma
For a normalized policy distribution:
```math
E_{A∼π_θ(·|s)}[∇_θ log π_θ(A|s)]=0
```
Proof:
```math
Σ_a π_θ(a|s)∇ log π_θ(a|s)
=Σ_a ∇π_θ(a|s)
=∇Σ_aπ_θ(a|s)=∇1=0
```
For continuous actions, replace the sum with an integral.

## 6. Baselines and variance reduction
A state-dependent baseline b(s) can be subtracted without changing the expected policy gradient, provided it does not depend on the sampled action:
```math
θ←θ+α[G_t−b(S_t)]∇_θlogπ_θ(A_t|S_t)
```
The baseline reduces variance by centering returns. A common choice is b(s)=Vπ(s).

A variance-minimizing scalar baseline for a given state and score vector is:
```math
b*(s)= E[G_t ||∇logπ||² |s] / E[||∇logπ||² |s]
```
The simpler conditional expected return E[G|s]=Vπ(s) is a common baseline, though not necessarily the exact variance-minimizing baseline in every parameterization.

## 7. Bias, variance and MSE
- Bias: Bias(θ̂)=E[θ̂]−θ.
- Variance: Var(θ̂)=E[(θ̂−Eθ̂)²].
- Mean squared error: MSE(θ̂)=E[(θ̂−θ)²] = Variance + Bias².

A lower-variance estimator is often more stable, but bias and variance must both be considered.

## 8. Advantage function
```math
A_π(s,a)=q_π(s,a)−v_π(s)
```
It measures how much better or worse action a is compared with the average value of state s.
- A>0: action better than average.
- A<0: action worse than average.
- A=0: equal to average.

Policy gradient with advantage:
```math
∇_θJ(θ)=E[∇_θlogπ_θ(A_t|S_t) A_π(S_t,A_t)]
```

## 9. Actor-Critic
Actor-Critic combines a policy learner (actor) and a value estimator (critic).

```text
       State S_t
          |
       ACTOR  ----> Action A_t
          |              |
          |              v
          |          Environment
          |              |
          |       Reward, next state
          |              |
          v              v
       Policy update  CRITIC
                         |
                      TD error
                         |
                  Feedback to actor
```

TD error:
```math
δ_t=R_{t+1}+γV_w(S_{t+1})−V_w(S_t)
```
For terminal next state, V(S_{t+1})=0.

Critic update:
```math
w←w+α_w δ_t ∇_w V_w(S_t)
```
Actor update:
```math
θ←θ+α_θ δ_t ∇_θlogπ_θ(A_t|S_t)
```
The critic estimates value and provides a learning signal; the actor adjusts action probabilities.

### Actor-Critic advantages
- Can update online at each step.
- Often reduces variance relative to plain Monte Carlo REINFORCE.
- Supports stochastic and continuous-action policies.
- Combines policy optimization with value estimation.

### REINFORCE vs Actor-Critic
| REINFORCE | Actor-Critic |
|---|---|
| Uses sampled return | Uses critic estimate / TD signal |
| No critic required | Actor and critic |
| Typically episode-based | Can be step-by-step |
| Often high variance | Often lower variance |

## 10. Policy gradient vs value-based methods
- Value-based: estimate V or Q, then derive actions (e.g. Q-learning).
- Policy-based: optimize πθ directly (e.g. REINFORCE).
- Actor-Critic: combines both ideas.
