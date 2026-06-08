# Blackjack RL Agent

Blackjack AI combining tabular Q-learning and DQN with card counting and a regression-based betting agent. Achieves +1.43% profit over the house edge using Hi-Lo counting and dynamic bet sizing. Built with Python & PyTorch.

---

## Table of Contents

- [Overview](#overview)
- [Phase 1 — Tabular Q-Learning](#phase-1--tabular-q-learning)
  - [Environment & MDP](#environment--mdp)
  - [Non-Counting Agent](#non-counting-agent)
  - [Card Counting Agent](#card-counting-agent)
  - [Phase 1 Results](#phase-1-results)
- [Phase 2 — Extended Environment](#phase-2--extended-environment)
  - [Task 1 — Tabular Q-Learning with Double-Down](#task-1--tabular-q-learning-with-double-down)
  - [Task 2 — DQN Playing Agent + Regression Betting Agent](#task-2--dqn-playing-agent--regression-betting-agent)
  - [Phase 2 Results](#phase-2-results)
- [Setup & Usage](#setup--usage)
- [Requirements](#requirements)

---

## Overview

This project explores reinforcement learning applied to the game of Blackjack across two phases of increasing complexity. The goal evolves from simply maximizing win rate (Phase 1) to maximizing profit through adaptive betting (Phase 2).

The environment is custom-built and model-free — the agent has no knowledge of transition probabilities and learns entirely by playing the game. Cards are dealt from a shuffled deck exactly as in a real Blackjack game, ensuring all states are encountered naturally during training.

---

## Phase 1 — Tabular Q-Learning

### Environment & MDP

The Blackjack environment is modelled as a **Markov Decision Process** with the following structure:

| Component | Description |
|---|---|
| State space | Player hand sum, dealer visible card, usable ace flag (+ count state for counting agent) |
| Action space | Hit, Stay |
| Transition probabilities | Unknown (model-free) |
| Reward function | +1 (win), 0 (draw), −1 (lose) |
| Discount factor | γ = 1 (no discounting) |

The deck is reshuffled at the start of every episode in the non-counting case. In the card counting case, reshuffling occurs only when fewer than 10 cards remain.

**State space sizes:**
- Non-counting agent: **200 states** (player sum × dealer card × usable ace)
- Card counting agent: **600 states** (3× due to Hi-Lo count state: Low / Neutral / High)

---

### Non-Counting Agent

**Algorithm:** Tabular Q-learning with ε-greedy exploration

| Parameter | Value |
|---|---|
| Training episodes | 2,000,000 |
| Learning rate (α) | 0.05 (constant) |
| ε start | 1.0 |
| ε end | 0.005 |
| ε decay | 0.99999 (linear) |
| Discount factor (γ) | 1 |

**Baseline policies (for comparison):**
| Policy | Win | Draw | Lose |
|---|---|---|---|
| Threshold-16 | 41.55% | 9.45% | 49.00% |
| Threshold-17 | 40.92% | 10.14% | 48.94% |

**Q-learning agent results** (seed=42, 100,000 test episodes):
| Win | Draw | Lose |
|---|---|---|
| 42.55% | 9.45% | 48.00% |

The learned policy correctly hits more aggressively when holding a usable ace, consistent with optimal Blackjack strategy. The ~42% win rate matches the theoretical ceiling for this game.

---

### Card Counting Agent

The Hi-Lo card counting system divides the state space into three count sub-states:
- **Low count** (< −3): fewer high-value cards remain → safer to hit
- **Neutral count** (−3 to +3): balanced deck
- **High count** (> +3): many high-value cards remain → higher bust risk

| Parameter | Value |
|---|---|
| Hi-Lo threshold | ±3 |
| Training episodes | 6,000,000 (3× larger state space) |
| ε decay | 0.999999 (slower, larger state space) |
| All other parameters | Same as non-counting agent |

**Results** (100,000 test episodes):
| Win | Draw | Lose |
|---|---|---|
| 42.35% | 9.48% | 48.17% |

**Per count-state breakdown:**
| Count State | Games | Win | Draw | Lose |
|---|---|---|---|---|
| Low (< −3) | 10,931 | **47.28%** | 9.09% | 43.63% |
| Neutral (−3 to +3) | 79,203 | 42.41% | 9.33% | 48.26% |
| High (> +3) | 9,866 | 40.39% | 8.87% | 50.74% |

In the Low count state, the agent actually beats the casino (wins more than it loses). The counting agent correctly learns to hit more in Low count states and hit less in High count states — exactly the desired behavior. However, overall win rate is similar to the non-counting agent, because with flat betting the advantage only becomes visible when combined with dynamic bet sizing (introduced in Phase 2).

---

### Phase 1 Results Summary

| Agent | Win | Draw | Lose |
|---|---|---|---|
| Non-counting Q-learning | 42.55% | 9.45% | 48.00% |
| Card-counting Q-learning | 42.35% | 9.48% | 48.17% |

Both agents converge to the theoretical optimal policy for Blackjack (~42% max win rate with flat bets).

---

## Phase 2 — Extended Environment

Phase 2 extends the environment with two new mechanics:
- **Natural Blackjack**: dealt 21 pays 1.5× the bet
- **Double-down action**: player can double the bet and receive exactly one more card

The goal shifts from maximizing win rate to **maximizing profit**.

---

### Task 1 — Tabular Q-Learning with Double-Down

The action space is expanded to {Hit, Stay, Double-down}. Two agents are trained: one without card counting and one with Hi-Lo card counting.

#### Without Card Counting

| Parameter | Value |
|---|---|
| Algorithm | Tabular Q-learning |
| Training episodes | 2,000,000 |
| α | 0.05 |
| ε decay factor | 0.999995 |
| ε range | 100% → 0.5% |
| γ | 1 |
| Shuffling | Every episode |

**Results** (100,000 test episodes):

| Win | Draw | Lose | Profit |
|---|---|---|---|
| 43.00% | 8.86% | 48.14% | **47.07%** |

Profit is calculated as:
```
profit = 50 - (1 - money / total_bet) * 100
```

The learned policy correctly applies double-down in favorable situations (e.g. player total of 10–11 against a weak dealer card), matching the well-known optimal strategy.

#### With Card Counting

| Parameter | Value |
|---|---|
| Hi-Lo threshold | ±3 |
| Training episodes | 6,000,000 |
| ε decay factor | 0.999999 |
| All other parameters | Same as above |
| Shuffling | When ≤ 10 cards remain |

**Results** (100,000 test episodes):

| Win | Draw | Lose | Profit |
|---|---|---|---|
| 42.86% | 9.29% | 47.85% | **47.44%** |

The policy adapts correctly per count state: hits more in Low count (fewer high cards left), hits less in High count (more high cards, higher bust risk). The Neutral count policy closely resembles the non-counting agent, as expected.

**Comparison to Phase 1:** The addition of natural Blackjack (1.5× payout) and double-down noticeably increases profit, even though flat win rates are similar.

---

### Task 2 — DQN Playing Agent + Regression Betting Agent

Task 2 separates the problem into two independent agents:
1. A **DQN-based playing agent** that decides hit/stay/double-down
2. A **regression-based betting agent** that decides how much to bet each round

#### Playing Agent

**Algorithm:** Deep Q-Network (DQN)

**Network architecture:**
- Fully connected
- ReLU activation
- Input layer → 2 hidden layers (128 neurons each) → output layer

**Inputs:**
| Feature | Description |
|---|---|
| Player hand sum | Current total card value |
| Usable ace | Boolean flag |
| Dealer card | Visible dealer card |
| Count | Integer value (−20 to +20) |

**Outputs:** Q-values for Hit, Stay, Double-down

| Parameter | Value |
|---|---|
| Learning rate | 1×10⁻⁴ |
| ε decay | 0.999975 |
| ε range | 100% → 0.5% |
| Batch size | 64 |
| Target network update | Every 1,000 episodes |
| Bust penalty | 1.1× |

**Results:**
| Deck | Win | Draw | Lose | Profit |
|---|---|---|---|---|
| 1 deck | 43.62% | 8.50% | 47.88% | 49.46% |
| 4 decks | 43.04% | 8.63% | 48.33% | 48.25% |

Using the raw integer count (−20 to +20) as input instead of the 3-state Hi-Lo discretization gives the DQN agent more information and improves performance over the Task 1 tabular agents.

---

#### Betting Agent

**Algorithm:** Regression neural network

**Network architecture:**
- Fully connected
- ReLU activation
- Input layer → 2 hidden layers (64 neurons each) → output layer

**Inputs:**
| Feature | Description |
|---|---|
| Count | Current Hi-Lo card count |
| Bet | Current bet size |
| Player advantage | `1 / (1 + exp(−0.5 × count))` |

**Output:** Expected mean reward for the given (count, bet, advantage) combination

**Training target:** Mean reward of all samples with the same (count, bet, player_advantage) — stabilizes training and guides the network to learn that higher counts warrant higher bets.

**Reward shaping during training:**
```
sample_reward = reward - 2 × (1 - player_advantage)² × bet
```
This penalizes high bets in unfavorable count states, teaching the network to bet conservatively when the deck is against the player.

**Learned policy:** The network correctly bets high when the count is high (player advantage) and bets the minimum when the count is low or neutral.

---

#### Combined Playing + Betting Agent Results

| Deck | Win | Draw | Lose | Profit |
|---|---|---|---|---|
| 1 deck | 43.62% | 8.50% | 47.88% | **51.43%** |
| 4 decks | 43.04% | 8.63% | 48.33% | **49.13%** |

The betting agent pushes profit above 50% on a single deck — **beating the house edge**. Win/draw/loss statistics are identical to the playing agent alone since the betting agent only affects stake size, not play decisions.

---

### Phase 2 Results Summary

| Agent | Profit (1 deck) | Profit (4 decks) |
|---|---|---|
| Task 1 — Q-learning (no counting) | 47.07% | — |
| Task 1 — Q-learning (card counting) | 47.44% | — |
| Task 2 — DQN playing agent only | 49.46% | 48.25% |
| Task 2 — DQN + betting agent | **51.43%** | **49.13%** |

---

## Setup & Usage

```bash
pip install -r requirements.txt
```

### Phase 1

```bash
# Train and test non-counting agent
python phase1/train_non_counting.py

# Train and test card counting agent
python phase1/train_counting.py
```

### Phase 2 — Task 1

```bash
python phase2/task1/train_tabular.py
```

### Phase 2 — Task 2

```bash
# Train playing agent
python phase2/task2/train_playing_agent.py

# Train betting agent (requires trained playing agent)
python phase2/task2/train_betting_agent.py

# Run combined evaluation
python phase2/task2/evaluate.py
```

---

## Requirements

```
torch
numpy
matplotlib
```

See `requirements.txt` for the full list.

---

## Authors

- Konstantinos Goutsias
- George Zervakis

Technical University of Crete — Reinforcement Learning Course Project
