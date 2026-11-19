# Important Questions – Unit-wise

Prepared from the important-question lists and syllabus supplied for the IV B.Tech I Semester CSE (Data Science) Reinforcement Learning course. These are revision priorities, not predictions of the exact exam paper.

## Unit I – RL Fundamentals and Probability
### Short answers
1. What makes RL different from other machine-learning paradigms?
2. State the Reward Hypothesis.
3. Define history Ht in RL.
4. Differentiate environment state and agent state.
5. Define a Markov state (information state).
6. What are the major components of an RL agent?
7. Distinguish deterministic and stochastic policies.
8. What does a value function predict?
9. What are model-free and model-based RL agents?
10. Differentiate learning and planning.
11. Explain exploration vs exploitation.

### Long answers
1. Explain the Markov property in detail and why it is important for RL.
2. Explain the RL agent–environment interaction and components.
3. Explain return, discount factor, policies and value functions.
4. Explain probability concepts: random variables, PMF, PDF, CDF, expectation, joint, marginal and conditional distributions, correlation and independence.

## Unit II – MRP, MDP and Dynamic Programming
### Short answers
1. Define return Gt in an MRP.
2. What does γ=0 mean?
3. Define state-value function v(s).
4. Write the Bellman equation for an MRP.
5. Define an MDP.
6. What is a policy π in an MDP?
7. State the Bellman optimality equation.
8. What are the two requirements for DP?
9. Define policy iteration and value iteration.
10. State Banach Fixed Point Theorem.
11. Define Bellman operator L.

### Long answers
1. Derive the Bellman equation for MRPs from the return.
2. Compare Markov processes, MRPs and MDPs.
3. Explain policy iteration with all steps and a flowchart.
4. Explain value iteration and its convergence properties.
5. Compare policy iteration and value iteration.
6. Explain how a policy transforms an MDP into an MRP.
7. Derive Bellman expectation equations for vπ and qπ.
8. Explain Bellman operator L and prove it is a contraction in max norm.
9. Explain Banach Fixed Point Theorem and its application to DP.
10. Prove iterative policy evaluation converges to the true value function.
11. Explain the principle of optimality and its role in DP.
12. Discuss asynchronous DP, transition matrices, small Gridworld and curse of dimensionality.

## Unit III – Monte Carlo and TD Learning
### Short answers
1. What is Monte Carlo policy evaluation?
2. Differentiate first-visit and every-visit MC.
3. Define on-policy and off-policy methods.
4. Define target and behaviour policies.
5. Write the importance-sampling weight.
6. Define TD learning and write TD(0) update.
7. What is the TD target?
8. Compare TD, MC and DP targets.
9. Define batch updating and exploring starts.
10. What do SARSA and Q-learning stand for / represent?

### Long answers
1. Explain first-visit and every-visit MC policy evaluation.
2. Explain on-policy vs off-policy learning and importance sampling.
3. Derive TD(0), TD target and TD error.
4. Compare DP, MC and TD.
5. Explain SARSA control algorithm with equation.
6. Explain Q-learning as off-policy TD control.
7. Compare SARSA and Q-learning using cliff walking.
8. Explain Expected SARSA and how it generalizes Q-learning.
9. Explain exploration in MC control and exploring starts.
10. Explain maximization bias and Double Q-learning.
11. Discuss convergence properties of TD methods.

## Unit IV – Function Approximation
### Short answers
1. What are the advantages of linear function approximation?
2. Name two neural-network architectures used for function approximation.
3. What objective is minimized in gradient descent?
4. Define the TD(0) target for semi-gradient TD.
5. What is the MC target used in Gradient MC?
6. Why is TD(0) called semi-gradient?
7. What is the TD fixed point?
8. What is experience replay in DQN?
9. What problem does experience replay solve?
10. What are afterstates?
11. Define LSTD and risk-sensitive RL.

### Long answers
1. Explain why function approximation is needed and limitations of tabular methods.
2. Explain linear function approximation, features and weight updates.
3. Derive SGD update for MSE in value function approximation.
4. Explain Gradient MC and semi-gradient TD(0), with equations.
5. Compare MC, TD(0) and TD(λ) with function approximation.
6. Explain neural networks as function approximators; compare with linear approximation.
7. Explain DQN architecture, training loop and hyperparameters.
8. Explain experience replay and target networks in DQN.
9. Explain LSTD, LSPI and LSTD-Q.
10. Discuss risk minimization approaches in RL.
11. Compare computational complexity of tabular, linear and neural approximation.

## Unit V – Policy Gradient Methods
### Short answers
1. What is the main goal of policy gradient methods?
2. State the log-derivative trick.
3. Write the REINFORCE update rule.
4. Compare naive REINFORCE and reward-to-go.
5. Define bias and mean squared error.
6. State the EGLP lemma.
7. Why does subtracting a baseline not change expected gradient?
8. Define the advantage function.
9. What are actor and critic?
10. Write the one-step actor-critic TD error.

### Long answers
1. Explain policy gradient methods and derive the log-derivative trick.
2. Explain REINFORCE with algorithm and update equation.
3. Compare naive REINFORCE and reward-to-go.
4. Explain EGLP lemma and prove that a state-dependent baseline does not change expected gradient.
5. Discuss bias, variance, MSE and variance reduction.
6. Derive the optimal baseline expression.
7. Explain the policy gradient expression using advantage function.
8. Describe actor-critic architecture and explain actor/critic updates.
9. Compare REINFORCE and actor-critic methods.

## Suggested priority
1. Unit II: Bellman equations, policy iteration, value iteration, contraction.
2. Unit III: MC vs TD, TD(0), SARSA, Q-learning.
3. Unit IV: function approximation, semi-gradient TD, DQN/replay.
4. Unit V: REINFORCE, baseline/EGLP, advantage, actor-critic.
5. Unit I: Markov property, return, policies, value functions and probability basics.

Prepare short answers as definitions + one formula where relevant. For long answers use: definition → explanation → equation → algorithm/diagram → example/advantages.
