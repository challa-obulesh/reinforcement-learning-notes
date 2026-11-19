# Unit I – Reinforcement Learning Fundamentals

## 1. What is Reinforcement Learning?
Reinforcement Learning (RL) is a learning paradigm in which an **agent** interacts with an **environment**, takes actions, receives rewards, and learns a policy that maximizes expected cumulative reward.

### Agent–environment interaction
```text
        action A_t
   Agent -----------> Environment
     ^                     |
     |                     |
     +---- state S_t, reward R_t+1
```

At time step t:
1. Agent observes state S_t.
2. Agent selects action A_t using policy π.
3. Environment returns reward R_(t+1) and next state S_(t+1).
4. Agent improves its behaviour from experience.

## 2. RL compared with other paradigms
- **Supervised learning:** learns from labelled input-output examples.
- **Unsupervised learning:** finds patterns in unlabelled data.
- **Reinforcement learning:** learns by interacting with an environment and receiving evaluative rewards; actions influence future observations.

## 3. Reward Hypothesis
All goals can be described as the maximization of expected cumulative reward. The reward signal defines what the agent should achieve.

## 4. Main components
- **Agent:** learner and decision maker.
- **Environment:** everything the agent interacts with.
- **State:** representation of the current situation.
- **Action:** decision taken by the agent.
- **Reward:** scalar feedback from the environment.
- **Policy π(a|s):** rule/probability distribution for selecting actions.
- **Value function:** expected return from a state or state-action pair.
- **Model:** predicts environment transitions and rewards, if available.

## 5. Return and discount factor
The return is the total discounted future reward:
```math
G_t = R_{t+1} + γR_{t+2} + γ^2R_{t+3}+... 
    = Σ_{k=0}^{T-t-1} γ^k R_{t+k+1}
```
Recursive form: **G_t = R_(t+1) + γG_(t+1)**.

- γ = 0: only immediate reward matters.
- γ close to 1: future rewards are important.
- Discounting helps handle delayed rewards and infinite-horizon sums.

## 6. History and Markov property
History:
```
H_t = S_0, A_0, R_1, S_1, A_1, R_2, ..., S_t
```
A state is Markov if it contains enough information to predict the future, so the future is conditionally independent of the past given the present state and action:
```math
P(S_{t+1},R_{t+1}|H_t,A_t)
= P(S_{t+1},R_{t+1}|S_t,A_t)
```
A **Markov state / information state** is a state representation satisfying this property.

### Environment state vs agent state
- Environment state: complete internal situation of the environment.
- Agent state: information available to the agent; it may be a full state or a summary of history.

## 7. Policies and value functions
- Deterministic policy: π(s) gives one action.
- Stochastic policy: π(a|s) gives a probability distribution over actions.

State-value function:
```math
v_π(s)=E_π[G_t | S_t=s]
```
Action-value function:
```math
q_π(s,a)=E_π[G_t | S_t=s,A_t=a]
```
A value function predicts expected future return, not just the immediate reward.

## 8. Model-free and model-based RL
- Model-free: learns values or policies directly from experience; does not require an explicit environment model.
- Model-based: uses or learns a model of transition probabilities and rewards for planning.

**Learning vs planning:** learning improves knowledge from interaction/data; planning computes decisions using a known or learned model.

## 9. Exploration vs exploitation
- Exploration: try actions to gain information.
- Exploitation: choose the action currently believed to be best.
A good agent balances both; too much exploration wastes reward, while too little can miss better actions.

## 10. Probability primer
- Random variable: numerical outcome of a random experiment.
- PMF (discrete): p(x)=P(X=x), with Σ_x p(x)=1.
- PDF (continuous): f(x)≥0 and integral f(x)dx=1; probabilities are areas.
- CDF: F(x)=P(X≤x).
- Expectation: E[X]=Σ_x xp(x) (discrete), or ∫xf(x)dx (continuous).
- Joint distribution: P(X=x,Y=y).
- Marginal: sum/integrate the joint distribution over the other variable.
- Conditional: P(X|Y)=P(X,Y)/P(Y), when P(Y)>0.
- Independence: P(X,Y)=P(X)P(Y).
- Covariance: Cov(X,Y)=E[(X−E[X])(Y−E[Y])]; correlation is normalized covariance.

## Quick revision
1. RL learns from interaction, not labelled answers.
2. Reward hypothesis: maximize expected cumulative reward.
3. Markov state summarizes all relevant past information.
4. Policy selects actions; value predicts returns.
5. Model-free learns from experience; model-based plans with a model.
