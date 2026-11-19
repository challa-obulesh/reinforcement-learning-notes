# Reinforcement Learning – Master Formula Sheet

Use this as a last-minute revision sheet. Symbols: s=state, a=action, r=reward, γ=discount factor, α=learning rate, π=policy.

## Unit I – Fundamentals
| Concept | Formula |
|---|---|
| History | Ht=(S0,A0,R1,...,St) |
| Markov property | P(S',R'|H,A)=P(S',R'|S,A) |
| Return | Gt=Σ(k=0 to T−t−1) γ^k R(t+k+1) |
| Recursive return | Gt=R(t+1)+γG(t+1) |
| State value | vπ(s)=Eπ[Gt|St=s] |
| Action value | qπ(s,a)=Eπ[Gt|St=s,At=a] |
| Policy | π(a|s)=P(At=a|St=s) |

## Unit II – MRP, MDP and DP
| Concept | Formula |
|---|---|
| MRP | (S,P,R,γ) |
| MDP | (S,A,P,R,γ) |
| MRP Bellman | v=R+γPv |
| Matrix solution | v=(I−γP)⁻¹R |
| MDP state value | vπ(s)=Σaπ(a|s)Σs'P(s'|s,a)[r+γvπ(s')] |
| MDP action value | qπ(s,a)=Σs'P(s'|s,a)[r+γΣa'π(a'|s')qπ(s',a')] |
| Optimal state value | v*(s)=maxa Σs'P(s'|s,a)[r+γv*(s')] |
| Optimal action value | q*(s,a)=Σs'P(s'|s,a)[r+γ maxa' q*(s',a')] |
| Policy improvement | π'(s)=argmaxa qπ(s,a) |
| Value iteration | Vk+1(s)=maxa Σs'P(s'|s,a)[r+γVk(s')] |
| Bellman contraction | ||Lv−Lu||∞≤γ||v−u||∞ |

## Unit III – Monte Carlo and TD
| Concept | Formula |
|---|---|
| MC incremental | V(s)←V(s)+α[G−V(s)] |
| Importance weight | ρt=π(At|St)/μ(At|St) |
| Trajectory weight | ρt:T−1=Πk=t to T−1 π(Ak|Sk)/μ(Ak|Sk) |
| TD target | R(t+1)+γV(S(t+1)) |
| TD error | δt=R(t+1)+γV(S(t+1))−V(St) |
| TD(0) | V(St)←V(St)+αδt |
| SARSA | Q←Q+α[r+γQ(s',a')−Q(s,a)] |
| Q-learning | Q←Q+α[r+γ maxa'Q(s',a')−Q(s,a)] |
| Expected SARSA | Q←Q+α[r+γΣa'π(a'|s')Q(s',a')−Q(s,a)] |

## Unit IV – Function Approximation
| Concept | Formula |
|---|---|
| Linear value estimate | v̂(s,w)=wᵀx(s) |
| MSE objective | J(w)=1/2 E[(target−v̂)²] |
| Gradient descent | w←w−α∇J(w) |
| Gradient MC | w←w+α[G−v̂(s,w)]∇v̂(s,w) |
| TD target | U=r+γv̂(s',w) |
| Semi-gradient TD | w←w+αδ∇v̂(s,w) |
| Linear semi-gradient TD | w←w+αδx(s) |
| DQN target | y=r+γ(1−done)maxa'Qtarget(s',a') |
| DQN loss | L=E[(y−Qonline(s,a;w))²] |
| LSTD | w=A⁻¹b |

## Unit V – Policy Gradients
| Concept | Formula |
|---|---|
| Objective | J(θ)=Eπθ[Gt] |
| Log derivative | ∇π=π∇logπ |
| REINFORCE | θ←θ+αGt∇logπθ(At|St) |
| EGLP | Eπ[∇logπθ(A|s)]=0 |
| Baseline | θ←θ+α(Gt−b(St))∇logπθ |
| Advantage | Aπ(s,a)=qπ(s,a)−vπ(s) |
| Actor-Critic TD error | δ=r+γV(s')−V(s) |
| Actor update | θ←θ+αθδ∇logπθ(a|s) |
| Critic update | w←w+αwδ∇Vw(s) |

## Common exam reminders
- γ=0: only immediate reward is counted.
- MC does not bootstrap; TD and DP bootstrap.
- SARSA uses the next action selected by the behaviour policy.
- Q-learning uses the maximum next-state action value.
- Policy gradient uses gradient ascent because expected reward is maximized.
- In a terminal transition, the next-state value term is zero.
