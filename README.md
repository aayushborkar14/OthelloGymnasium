<img src="othello_ai/imgs/ScreenHumanAnsi.png">  

# Gymnasium Othello Environment

This repository contains an implementation of OTHELLO - often also called REVERSI - with OpenAI [Gymnasium](https://gymnasium.farama.org/index.html) interfaces. This environment is for researchers and engineers who are interested in developing model-based Artificial Intelligence Reinforced Learning algorithms.

Several simple ready baselines are provided:
1. Random policy (random moves)
2. Greedy policy (max number of opponent's pieces captured)
3. [Maximin](https://en.wikipedia.org/wiki/Minimax) policy
4. Human policy (manual move)
which may be used to experiment with RL algorithms.

## Basic Usage

```
# -- Othello - activation examples --

# Play against machine, opponent default Rand policy
python run_ai.py --protagonist='human'

# Play against machine with a reduced board size, opponent default Rand Policy
python run_ai.py --protagonist='human' --board-size=6

# Play against machine, which use maximin strategy, with a reduced board size
python run_ai.py --protagonist='human' --render-mode='human' --opponent='maximin' --board-size=6 --rand-seed=100

# Play against machine, which use random strategy, with a reduced board size
python run_ai.py --protagonist='human' --render-mode='human' --opponent='rand' --board-size=6 --rand-seed=100

# Play against machine, which use Greedy strategy, with a reduced board size
python run_ai.py --protagonist='human' --render-mode='human' --opponent='greedy' --board-size=6

# Play against machine - Greedy -, a reduced board size and ansi representation
python run_ai.py --protagonist='human' --render-mode='ansi' --opponent='greedy' --board-size=6

# Train machine, which use rand strategy, with render mode ansi
python run_ai.py --protagonist='rand' --render-mode='ansi'

# Train machine, which use maximin strategy, with a reduced board size
python run_ai.py --protagonist='maximin' --opponent='rand' --board-size=6 --rand-seed=100

```

## Local Installation
### Create a virtual env based on Python 3.11
Install Python 3.11 - GYMNASIUM most recent python version supported - in your local machine, if not existent, and create a virtual env:
```
mkdir gym
python3.11 -m venv gym
cd gym
source bin/activate
```
### Install GYMNASIUM and related classic controls
```
pip install gymnasium[classic-control]
```
### Download the OthelloGymnasium Package and install it locally
Browse the url https://github.com/pghedini/OthelloGymnasium and download zip file. 
Unpack the zip file in your working directory. 
```
unzip OthelloGymnasium-main.zip
pip install -e OthelloGymnasium-main
```
### Run the start.sh bash script in order to try the installed package
```
cd OthelloGymnasium-main/othello_ai
bash start.sh
```
## Use the package in interactive mode
To use the package in interactive mode:
- run the python interpreter in your virtual env and execute...
```
import gymnasium
import othello_ai
env = gymnasium.make('othello_ai/Othello-v0', render_mode='ansi', board_size=6)
# start a new game
env.reset()
# get the color next to move
next_move = lambda x: "BLACK" if x == -1 else "WHITE"
next_move(env.unwrapped.env.unwrapped.player_turn)
...
```
For a complete usage example, inspect the code in run_ai.py.

## Citation
This work is based on the code by Lerry Tang, which can find at https://github.com/lerrytang/GymOthelloEnv.
The code has been profoundly rewritten to adapt it to recent versions of [GYMNASIUM](https://gymnasium.farama.org/index.html) and to provide it with a [PYGAME-based](https://www.pygame.org) graphic interface.

```
@misc{othellogymnasium,
  author = {Pierfrancesco Ghedini},
  title = {OthelloGymnassium},
  year = {2024},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/pghedini/OthelloGymnasium}},
}
```
