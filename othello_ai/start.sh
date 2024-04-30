#
# -- AI Othello - activation examples --
#
# Play against machine, opponent default Rand policy
#python run_ai.py --protagonist='human'
#
# Play against machine with a reduced board size, opponent default Rand Policy
#python run_ai.py --protagonist='human' --board-size=6
#
# Play against machine, which use maximin strategy, with a reduced board size
#python run_ai.py --protagonist='human' --render-mode='human' --opponent='maximin' --board-size=6 --rand-seed=100
#
# Play against machine, which use random strategy, with a reduced board size
#python run_ai.py --protagonist='human' --render-mode='human' --opponent='rand' --board-size=6 --rand-seed=100
#
# Play against machine, which use Greedy strategy, with a reduced board size
#python run_ai.py --protagonist='human' --render-mode='human' --opponent='greedy' --board-size=6
#
# Play against machine - Greedy -, a reduced board size and ansi representation
python run_ai.py --protagonist='human' --render-mode='ansi' --opponent='greedy' --board-size=6
#
# Train machine, which use rand strategy, with render mode ansi
#python run_ai.py --protagonist='rand' --render-mode='ansi'
#
# Train machine, which use maximin strategy, with a reduced board size
#python run_ai.py --protagonist='maximin' --opponent='rand' --board-size=6 --rand-seed=100
