# Checkers 6x6 PettingZoo Environment

This repository contains a custom PettingZoo AEC environment for 6x6 Checkers and a self-play Actor-Critic training pipeline.

## Environment Name

- `checkers_6x6_v0`
- File: `mycheckersenv.py`

## Agents

- `player_0`
- `player_1`

Agents move in strict AEC turn order.

## Observation Space

Each agent receives:

- `board`: `Box(low=0, high=1, shape=(6, 6, 4), dtype=int8)`
  - Channel 0: current agent's men
  - Channel 1: current agent's kings
  - Channel 2: opponent men
  - Channel 3: opponent kings
- `action_mask`: `Box(low=0, high=1, shape=(288,), dtype=int8)`
  - `1` marks legal actions and `0` marks illegal actions
  - mandatory capture is enforced through this mask

## Action Space

- `Discrete(288)`
- Action encoding:
  - board has 36 source squares
  - each source has 8 directional templates
    - simple moves: NW, NE, SW, SE
    - jump moves: NW, NE, SW, SE (two-cell diagonal capture)

Encoded as:

- `action = from_square * 8 + direction_code`
- `from_square = row * 6 + col`

## Game Rules Implemented

- 6x6 board
- diagonal movement
- kings supported
- mandatory captures
- **multi-jump**: after a capture, if the same piece can jump again from its landing square, it must do so in the same turn (same agent again) until no further jump is available from that piece
- promotion when a man reaches the far row
- win if opponent has no pieces
- win if opponent has no legal moves

## Reward Structure

- win: `+1`
- loss: `-1`
- illegal move by acting agent: acting agent `-1`, opponent `+1`, game terminates
- non-terminal regular move: `0`
- truncation draw (max move limit): `0` for both

## Termination and Truncation

Episode terminates when:

- one side has no remaining pieces
- next player has no legal moves
- illegal move is played

Episode truncates when:

- move count reaches `200`

## Training Files

- `myagent.py`: Actor-Critic network and update rule
- `myrunner.py`: self-play training and sample run rendering

Run training and sample game:

```bash
python3 myrunner.py
```