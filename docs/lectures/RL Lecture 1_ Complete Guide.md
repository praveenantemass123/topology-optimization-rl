UCL / DeepMind · Reinforcement Learning course · Lecture 1 by Hado van Hasselt

# Introduction to Reinforcement Learning: Complete Guide

One structured walk-through of the lecture: motivation, the framework, the mathematics, the agent’s components, and worked examples. Companion text: Sutton & Barto, *Reinforcement Learning: An Introduction*.

**In one line:** RL is the science and framework of learning to make decisions from interaction. An agent acts, observes and receives a scalar reward, and learns to maximize cumulative future reward.

[1. Course context and logistics](#s1)[2. What is RL? Motivation from AI](#s2)[3. The agent–environment loop](#s3)[4. Reward and the reward hypothesis](#s4)[5. Where RL applies, and two reasons to learn](#s5)[6. Return, value and action value](#s6)[7. The discount factor](#s7)[8. What is inside an agent](#s8)[9. State, history and the Markov property](#s9)[10. Partial observability and the maze](#s10)[11. Policies](#s11)[12. Value functions and the Bellman equations](#s12)[13. Models of the environment](#s13)[14. Agent categories](#s14)[15. Prediction, control, learning and planning](#s15)[16. Deep reinforcement learning](#s16)[17. Worked examples](#s17)[18. Course roadmap and takeaways](#s18)

## 1. Course context and logistics

Moodle is the communication portal; ask questions on its forum so everyone benefits. Grading is by assignments, with no exam this year. Lectures were pre-recorded because of COVID-19.

## 2. What is RL? Motivation from AI

### The big-picture argument

- **Industrial revolution:** automating repeated *physical* work (steam trains replacing hauling carts).
- **Digital revolution:** automating repeated *mental* work (a calculator runs addition and multiplication that we already know how to do).
- In both, humans first had to *find the solution*. The next step is letting **machines find solutions themselves**. We specify a problem and a goal, and the machine works out how to solve it. That is AI, and it involves **learning**, **autonomy** and **decisions**.

### Alan Turing, 1950

In “Computing Machinery and Intelligence” Turing asked “can machines think?” and suggested that instead of programming an adult mind, we should program a *child’s* mind (little mechanism, many blank sheets) and subject it to education. Writing a program that can learn may be easier than writing one with all the adult’s skills. The lecturer invites you to ponder whether you agree.

**Working definition of AI used here:** the ability to **learn to make decisions to achieve goals**. It is not the only definition, but it is the central question of the course.

### How RL learning differs from other learning

- **Active, not passive:** your actions influence the data you see.
- **Sequential:** later interactions depend on earlier ones.
- **Goal-directed:** behavior has a purpose, from picking up a glass down to individual muscle signals.
- **No examples of optimal behavior:** nobody tells you the exact low-level actions (muscle commands). You learn by optimizing a reward signal such as satisfaction.

### Three ways to get a machine to do a task

Think of three ways to get a machine to do a task. **(a) Hand-code it:** a human works out the solution and writes it down (a calculator). **(b) Supervised learning:** a human supplies the correct answer for many examples. **(c) Reinforcement learning:** a human supplies only a *goal* (through a reward) and the machine discovers the behavior by trying things.

Why is (c) powerful? Nobody can write down the exact muscle commands for riding a bicycle, or the best move in every Go position. The lecture’s point is that most human learning is like this: someone may show you a high-level example (“write like this”) but never the low-level signals needed to carry it out. RL targets exactly that gap.

## 3. The agent–environment loop

An **agent** interacts with an **environment**. The agent sends **actions**; the environment returns **observations** (and rewards). The agent may be a robot in the real world, or a program in a game or on the internet. The agent’s sensors (for example a camera) define what it observes.

at time t: agent receives O_t, R_t agent emits A_t environment returns O\_(t+1), R\_(t+1)

The time index increments after the action is taken (convention). RL can be extended to continuous time, but this course uses discrete steps. The reward can be viewed as coming from the environment or as an internal preference over observations; either way a goal must be clearly specified.

## 4. Reward and the reward hypothesis

The claim: **any goal can be expressed as maximizing the cumulative sum of a scalar reward.** Three things are often misread:

- **It says a reward exists, not that it is easy to find.** Designing a reward that is both correct and easy to optimize is a real research problem (reward design).
- **Cumulative, not immediate.** “Maximize the next reward” is a much weaker goal than “maximize the total.”
- **Timing can be encoded.** Compare two rewards for “reach the exit”:

| Design | Reward | Resulting behavior |
| --- | --- | --- |
| Sparse goal | 0 each step, +1 on reaching the exit | Any path that eventually arrives scores the same (without discounting) |
| Step penalty | −1 each step, 0 after the exit | Fewest steps wins, because each extra step costs 1 |

Same goal, different incentives, and the second says *how fast* as well as *whether*. In the Atari demo the reward was simply the change in score, so most steps give 0 and some give positive values.

## 5. Where RL applies, and two reasons to learn

Examples that RL has been successfully applied to: flying a helicopter (reward tied to air time or distance to a goal), managing an investment portfolio, controlling a power station, making a robot walk, and video and board games (+1 for a win, −1 for a loss in chess).

**Important distinction:** the RL *problem* (learning via interaction with rewards) is separate from today’s set of *solution algorithms*. If your problem has reward, sequential interaction and learning, you are doing RL, whether or not your algorithm is labelled “RL”. Be flexible about solutions.

| Reason to learn | Idea | Examples |
| --- | --- | --- |
| Find a solution | Learn a policy, then deploy it. May include indirect steps (refuel first). | Strong chess program; manufacturing robot |
| Adapt online | Keep learning to handle unforeseen circumstances (unknown unknowns), not just to generalize widely. | Chess opponent that wins about half the time or keeps you playing; robot facing new terrain or wear and tear |

A small adaptive learner that tracks the world can be easier than one giant solution covering every possible case.

### So, what is RL?

RL is the **science and framework of learning to make decisions from interaction**. It requires thinking about time and consequences, actively gathering experience, predicting the future and handling uncertainty. This makes it harder than, say, plain classification, but its scope is huge: it formalizes the AI problem. It works closely with deep learning.

### Atari demo

One agent learned games such as Beam Rider and Enduro (the algorithm is **DQN**, covered later). Observations are raw **pixels**; actions are **joystick inputs** (directions and fire); reward is the **change in score** each step. The agent is not told what it controls or any strategy, and the same generic algorithm learns very different games.

## 6. Return, value and action value

### Return

Suppose after time t the rewards are 0, 0, +1, 0, +10 and the episode ends. The (undiscounted) return is 11. The return counts only the future, because actions at time t cannot change rewards already received.

### Why expectation?

The world and the agent’s choices may be random, so the return itself is random. The **value** is its average over that randomness: v(s) = E\[G_t | S_t = s\]. It answers “how good is it, on average, to be here?”

### Recursive view, with numbers

The return splits as G_t = R\_(t+1) + γ G\_(t+1). Take γ = 0.9 and rewards 0, 0, 10 afterward:

G at last step before reward = 10 G one step earlier = 0 + 0.9 × 10 = 9 G two steps earlier = 0 + 0.9 × 9 = 8.1

This is why values “shrink” with distance from a reward. It is also why the recursion is so useful: to know the value of now, you only need the immediate reward plus the (discounted) value of the next state.

### State value vs action value

**v(s)** averages over whatever the policy would do. **q(s,a)** fixes the first action to a, then follows the policy. If you know q for all actions, choosing the action with the largest q gives a better policy. This is the basis of Q-learning and DQN later.

v_π(s) = Σ_a π(a|s) · q_π(s,a)

### Delayed reward and credit assignment

A reward at time t may be caused by an action many steps earlier. This is the **credit assignment** problem. Values handle it by passing reward information backward: a state that leads (even indirectly) to a big reward gets a big value, so actions leading there look good. Refueling a helicopter earns nothing now or even costs reward, but it keeps the larger future rewards reachable, so it has high value.

## 7. The discount factor

G_t = R\_(t+1) + γ R\_(t+2) + γ² R\_(t+3) + ... 0 ≤ γ ≤ 1

| γ | Reward 10 steps away is worth | Agent character |
| --- | --- | --- |
| 0 | 0 | Myopic: only the next reward |
| 0.5 | ≈ 0.001 of its size | Very short-sighted |
| 0.9 | ≈ 0.35 | Plans a few tens of steps ahead |
| 1 | full value | Far-sighted; can make sums infinite in non-ending tasks |

Two roles: (1) a **mathematical convenience** that keeps infinite sums finite (with +1 per step forever, γ = 1 gives infinity); (2) a **goal-shaping parameter**. With “0 until the goal, then +1” and γ \< 1, reaching the goal sooner gives a larger value, so the agent prefers short paths. The effective horizon is roughly 1/(1−γ) steps.

## 8. What is inside an agent

The framework has an **environment** (dynamics), a **reward signal** (the goal) and an **agent**. The agent contains:

- **Agent state** (required): everything carried from one step to the next.
- **Policy** (required): how actions are selected (even uniformly random counts).
- **Value function estimate** (optional but common).
- **Model** (optional): a prediction of the environment’s dynamics.

## 9. State, history and the Markov property

Use a concrete picture: you playing a video game.

- **Environment state:** everything the game engine stores (all enemy positions, hidden timers, random-number seeds). You never see it all.
- **Observation:** the screen you see right now.
- **History:** every screen, button press and score change since you started.
- **Agent state:** your mental summary: “enemy came from the left, I have two lives, power-up active.” It is what you carry forward, and it should be built from history only.

The agent state is defined by an update rule: new state = u(old state, action, reward, new observation). Because of that, the agent never needs to store the whole history, only a running summary.

### Full observability

If the observation *is* the environment state, the agent state can simply be the observation and no memory is needed. This is the special case that theory often assumes because it is easy to reason about, but it is rare in practice.

**Meaning:** “the future depends on the past only through the present state.” If you know S_t, the rest of the history adds no predictive information about the next reward and next state.

p(r, s' | S_t = s) = p(r, s' | H_t) (probability of the next reward r and next state s' given the state equals that given the whole history)

Examples: a chess position is Markov (the board tells you everything needed to continue; how you got there does not matter, ignoring special rules such as repetition and castling rights). The velocity-free screenshot of a ball in Pong is **not** Markov, because you cannot tell which way the ball is moving, so you need at least two frames. That is exactly why Atari agents stack frames.

### Why it matters

Algorithms with the Markov assumption (value recursion, Bellman equations, MDP theory) only work cleanly if the state really summarizes the past. If it does not, the agent confuses different situations that look identical, which is called **perceptual aliasing**.

## 10. Partial observability and the maze

In a **POMDP** the agent sees an observation that is only a slice of the true state. The lecture’s maze: the agent sees just the 3×3 cells around itself. Two corridor spots can produce the same 3×3 view.

1. Goal: top-left. Start: top-right.
2. At the start, shortest path to the goal means going **down**.
3. At the second, look-alike spot, shortest path means going **up**.
4. A policy that is a function of the current observation must output one action for the identical view, so it must be wrong in at least one place.

### Candidate fixes

| Agent state | Works? | Issue |
| --- | --- | --- |
| Current observation only | No | Aliasing, as shown |
| Full history | Yes (always Markov) | Grows without bound |
| Last two observations | Sometimes | Fails with random policy or noisy moves; works if the policy never walks between the ambiguous cells in the confusing order |
| Learned recurrent summary | Often well enough | Needs to be learned; approximate |

The moral: you rarely get a perfect Markov state, and you do not need one. What you need is a state good enough to support good decisions and value estimates.

### The state update function

S\_(t+1) = u(S_t, A_t, R\_(t+1), O\_(t+1))

This recursive form covers every case above: concatenating everything gives the full history, keeping only the newest observation gives “state = observation”, and a two-slot buffer drops the oldest observation each step. We want *u* to compress the stream into a fixed-size state that keeps what matters for acting.

## 11. Policies

A **policy** maps agent state to action: deterministic `a = π(s)`, or stochastic `π(a|s)`, a probability distribution over actions. It is a function of the *agent* state, never of hidden environment state.

### Why stochastic policies?

- They give built-in **exploration**: trying actions that currently look worse in order to learn more (the topic of the next lecture).
- They are smooth to optimize: small changes to probabilities give small changes in behavior, which is what policy-gradient methods use.
- They include the deterministic case as a special case.

### What “value of a policy” means

Writing v_π conditions on a *function*, not a fixed list of actions: at every future step the action is chosen by π after seeing the state reached, which may differ from run to run. Two policies can then be compared state by state, and a better one can be built from the values of the current one (**evaluate, then improve**). Later lectures turn this loop into real algorithms.

## 12. Value functions and the Bellman equations

Start with v_π(s) = E\[G_t | S_t = s\] and substitute G_t = R\_(t+1) + γ G\_(t+1):

v_π(s) = E\[ R\_(t+1) + γ v_π(S\_(t+1)) | S_t = s \]

Read it as: **value now = expected immediate reward + discounted value of where I land.** This is a consistency condition: the values of neighbouring states must agree with each other.

### Tiny example

States X → Y → Terminal. Reward +1 on X→Y, +2 on Y→Terminal, γ = 0.9, one action only.

v(Terminal) = 0 v(Y) = 2 + 0.9 × 0 = 2 v(X) = 1 + 0.9 × 2 = 2.8

### The optimality version

v\*(s) = max_a E\[ R\_(t+1) + γ v\*(S\_(t+1)) | S_t = s, A_t = a \]

The only change is “max over actions” instead of “follow policy π.” Because the unknown v\* appears on both sides, algorithms can start with a guess, repeatedly apply the right-hand side, and converge to v\*. That idea, fixed-point iteration on the Bellman equation, is behind dynamic programming, Q-learning and DQN in later lectures.

### From values to behavior

Given q\*, the optimal policy is just: in each state take the action with the highest q\*. This is why a good value function is so valuable, and also why approximate values can still give very good policies.

## 13. Models of the environment

### Model

A model predicts what the environment does next: a transition model p(s'|s,a) and a reward model E\[R|s,a\]. Unlike a value function, a model does not directly give a policy; you need **planning** on top. Model variants include distribution models (full distribution), expectation models, and sample models (a plausible next state). The right choice is not settled.

A **value function** says how good things are; it hides how the world works. A **model** says how the world works; to get behavior from it you still must plan (search or simulate). Value-based methods act immediately but can’t answer “what if”; model-based methods can imagine futures but depend on an accurate model. In the lecture’s maze example a learned model missed an unexplored corner, so planning inside it can only solve what it has seen.

### Simple maze illustration

Start at the left, goal at the right, reward −1 per step, actions N/E/S/W, state = position (fully observable). The optimal policy is the shortest path; its value rises by 1 in magnitude for each step further from the goal (−1, −2, and so on). A learned model can be imperfect, for example missing part of the maze it has not explored; policies and values can also be imperfect.

## 14. Agent categories

Two independent questions classify an agent: **what does it learn** (value, policy, both) and **does it have a model**?

| Type | Components | How it acts | Notes |
| --- | --- | --- | --- |
| Value-based | Value function only | Policy read off values, e.g. pick the action with the highest q (mostly) | The Atari DQN agent |
| Policy-based | Explicit policy only | Samples from π directly | Learns without values; example next lecture |
| Actor-critic | Policy (actor) + value (critic) | Actor acts, critic scores its choices and guides updates | Common in deep RL |
| Model-free | Policy and/or value, no dynamics model | Acts from learned values/policy | Simple; needs plenty of real experience |
| Model-based | Dynamics model, optionally policy/value | Plans with the model (or mixes planning with a policy) | Can “imagine”; limited by model accuracy |

A value function does model something (expected cumulative reward), but in RL vocabulary “model” means a model of the environment’s dynamics, so a pure value agent is called model-free. These labels are conventions, not a perfect division.

## 15. Prediction, control, learning and planning

### Prediction vs control

**Prediction** evaluates the future for a given policy (for instance, the value of the uniformly random policy). **Control** optimizes the future by finding the best policy. They are linked: π\* is the policy whose value is highest, so values rank policies, and good predictions let you improve control.

### Learning vs planning

|  | Learning | Planning |
| --- | --- | --- |
| Input | New experience from the interaction loop | A model (given or learned); no new experience |
| Nature | Absorbs data into values, policy or model | Internal computation that improves policy or predictions |

Both can run together in one agent. **Caveat:** planning in a *learned* model can exploit its errors. If the model wrongly believes a wall has a hole, the shortest path goes through that hole and the resulting policy fails in the real world.

### Is prediction enough?

Sutton often argues that predictions are a rich form of knowledge: if you could predict everything, what else would you need? The lecture’s hint is that with imperfect approximation it still helps to store predictions and policies (skills) separately, since knowing the world does not automatically tell you what to do.

## 16. Deep reinforcement learning

Every component is a function: policy (state→action probabilities), value (state→expected return), model (state, action→next state, reward), and the state update *u*. Neural networks are strong function learners, and combining them with RL is **deep RL**.

### Why it is not plug-and-play

- **Correlated data:** a robot in one room produces strongly similar consecutive samples, while supervised learning assumes roughly independent examples.
- **Non-stationary targets:** when the policy changes, the value function of the *current* policy changes with it, so the learning target keeps moving.

Later lectures cover the remedies; the point here is that deep RL is a research area of its own at the intersection of the two fields.

## 17. Worked examples

### Atari (DQN)

| Element | In Atari |
| --- | --- |
| Observation | Raw screen pixels |
| Action | Joystick directions plus fire |
| Reward | Change in score each step (mostly 0, sometimes positive) |
| Prior knowledge | None: not told what the player is, the rules or any strategy |

Because the algorithm is generic, the same agent learns very different games. It is a value-based, model-free agent.

### The 5×5 grid world (Sutton & Barto), γ = 0.9

- Bumping into a wall: −1. Most other steps: 0.
- Any action in **A**: +10 and teleport to **A′**. Any action in **B**: +5 and teleport to **B′**.

**Prediction (uniform random policy).** v(A) ≈ 8.8, below 10 because after the +10 the random agent soon bumps into walls and collects −1 penalties. v(A′) ≈ −1.3 since A′ sits at the edge and the walk back toward A is unproductive. Bottom-left states are strongly negative because a random agent repeatedly hits walls.

**Control (optimal policy).** All values become positive since an optimal agent never bumps into a wall. Even though B is closer to B′, looping A → A′ → … → A pays +10 per loop, which beats the +5 loop of B, so almost every state heads to A. Only the top-right corner jumps into B, because walking around B to reach A would take longer than being teleported to B′ and walking on to A. These conclusions are hard to see by inspection, but learning and planning algorithms find them automatically.

### Learned locomotion

Simulated 2D and 3D bodies learned to run, jump and climb obstacles with one reward: speed in a single direction. They sensed the terrain through simplified vision and proprioception (sensing their own limbs). No gait was hand-designed, and the same learner worked across body types and terrains.

## 18. Course roadmap and takeaways

1. Exploration and **bandits** (one-step decision problems), next lecture.
2. **Markov decision processes** and planning with **dynamic programming** (Diana Borsa).
3. **Model-free prediction and control**, including Q-learning and DQN (deep Q-network, Q-learning plus deep nets).
4. **Policy-gradient** and **actor-critic** methods.
5. **Deep RL**.
6. **Integrating learning and planning**.

The emphasis is on core principles rather than current state-of-the-art algorithms, which will change, while principles let you understand and invent new ones.

**Key takeaways:**

- RL is learning to decide from interaction, to maximize cumulative reward.
- Any goal can (in principle) be expressed as a reward.
- The agent has state, a policy, and optionally values and a model.
- Most real problems are partially observable, so the agent state must be built from history.
- Value functions (and Bellman equations) are the main tool for turning prediction into good control.
- Separate the RL *problem* from any particular *algorithm*.

**Check yourself:**

1. Why can γ = 1 be a problem in a task that never ends?
2. Give an observation that is not Markov in a game you know, and a minimal fix.
3. Rewrite v(s) in terms of q(s,a) and π.
4. Why does the Bellman optimality equation not mention a policy?