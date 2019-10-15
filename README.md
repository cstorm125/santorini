# SantoriniGo
Reinforcement Learning Agents to Play The Board Game Santorini

## Game Rules
We slightly modified the original [Santorini](https://boardgamegeek.com/boardgame/194655/santorini) boardgame to fit our training regime a little bit better.

### Setup
* 5x5 grid
* 2 players (modified from 2-4 players)
* 2 workers each
* Building pieces
    * 22 First floors
    * 18 Second floors
    * 14 Third floors
    * 18 Domes
* Workers start across from each other (modified from anywhere on the board)
* No special powers (modified from special powers randomly given by cards)

### Each Turn
* Player chooses one of their workers to move to a cell on the board that is:
    * Adjacent to the current position in all eight directions
    * Empty; no other workers standing or no dome covering it
    * No more than one level higher than the current position
* Using that worker at the new current position, the player then choose to build on a cell that is:
    * Adjacent to the current position in all eight directions
    * Building progression is as follows: Empty Floor -> First Floor -> Second Floor -> Third Floor -> Dome
    * A worker cannot build if there are no parts left
    
### Win Condition
The player whose any of their workers stand on the third floor automatically wins.

## Environment

### Initialization

Initialize the game with 2 players: `-1` and `1`. `-1` has negative workers `-1` and `-2` whereas `1` has positive workers `1` and `2`. `-1` always starts first.

```
from santorinigo.environment import Santorini
env = Santorini()
```

### State

In order to see the board as a spectator,

```
> env.print_board()
Buildings:
 [[1 0 0 0 0]
 [0 0 0 0 0]
 [0 0 0 0 0]
 [0 1 0 0 0]
 [0 0 0 0 0]]
Workers:
 [[ 0 -1  0  0  0]
 [ 0  0  0  0  0]
 [ 0  0  0  0  2]
 [ 1  0  0  0  0]
 [ 0  0 -2  0  0]]
Parts:
 [[ 0  0  0  0  0]
 [ 0 20  0  0  0]
 [ 0  0 18  0  0]
 [ 0  0  0 14  0]
 [ 0  0  0  0 18]]
```

In order to get state to train the model (negative workers will belong to current players),

```
> env.get_state(flattened=False)
array([[[ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0]],

       [[ 0,  0, -1,  0,  0],
        [ 0,  0,  0,  0,  0],
        [ 1,  0,  0,  0,  2],
        [ 0,  0,  0,  0,  0],
        [ 0,  0, -2,  0,  0]],

       [[ 0,  0,  0,  0,  0],
        [ 0, 22,  0,  0,  0],
        [ 0,  0, 18,  0,  0],
        [ 0,  0,  0, 14,  0],
        [ 0,  0,  0,  0, 18]]])
```

The environment return a state as a three-dimensional tensor (`numpy.array`) each containing a 5x5 tensor:
1. All buildings on the board denoted 0 to 4
2. Positions of all workers; workers of current player will always be negative if you use `env.get_state()`
3. Number of remaining building parts on the diagonal

If `flattened=True`, it will return a flattened `numpy.array` with 55 dimensions summarizing 1., 2. and 3.

```
> env.get_state(flattened=True)
array([ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
          0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
          0,  0,  0,  0,  0, -2, -1,  0,  0,  0,  0,  0,  0,  2,  0,  0,  0,
         20, 18, 14, 18])
```

### Action

The environment has 128 unique actions which are combinations of:
1. Worker selection: `-1` or `-2` (workers of current player will be treated negative when choosing actions)
2. Move direction: 'q', 'w', 'e', 'a', 'd', 'z', 'x', 'c'; directions are based on the keyboard with 's' as the center
3. Build direction: 'q', 'w', 'e', 'a', 'd', 'z', 'x', 'c'

```
> env.itoa #integer to action
[(-1, 'q', 'q'),
 (-1, 'q', 'w'),
 (-1, 'q', 'e'),
 (-1, 'q', 'a'),
 (-1, 'q', 'd'),
 (-1, 'q', 'z'),
 (-1, 'q', 'x'),
 (-1, 'q', 'c'),
 ...]
```

You can check if the action is legal by,

```
> env.legal_moves()
[27, 28, 29, 30, 31, 35, 36, ...]
```

### Reward
The agent will get `+1` reward for winning and `0` otherwise. Each move can have a penalty by adjusting `move_reward = -1e-3` in `env.step()`. 

### Play

You can play the environment as two random agents who will only do legal moves like this:

```
actions = np.arange(128)
while True:
    #select action
    np.random.shuffle(actions)
    #usually choose players but in this case it doesn't matter because we are doing random stuff now
    if i % 2 == 0: 
        actions = actions
    else:
        actions = actions

    #check legality
    legal_moves = env.legal_moves()
    for a in actions:
        if a in legal_moves:
            action = a
            break

    #step action
    next_state,reward,done,next_player = env.step(action, move_reward = 0)

    #break if done
    if done: break
```