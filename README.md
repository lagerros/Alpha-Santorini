# Alpha-Santorini
In short: Alpha Go Zero applied to the game Santorini. Work in progress.

## The project
I replicated AlphaGo Zero from scratch in an attempt to get hands-on experienced with deep learning. I chose to apply it to the game of Santorini in order to create an unhackable challenge. Santorini has never been solved (due to limited attempts, rather than intrinsic difficulty!), so there would be no possibility of me, as a human, overfitting my own learning, for example by extensively copying someone else’s code. The only way of solving the challenge was by _actually doing it_. If we want to optimize, we should practice what we preach! :) 

## The components
master.py coordinates self-play, training, and evaluation, and has options for serial and parallel modes. This project is work in progress and there might be some residual issues, especially with the parallelization, but I expect it to work “out of the box”.

nets.py contains the class used to initialize the convolutional/residual network. 

M.py contains the algorithm for building the search tree data structure and traversing it by Monte Carlo search.


## ...but will it train?
AlphaGo Zero is an incredibly compute intensive project, and trying to train to superhuman level on a laptop using this code would take thousands of years. Yet hobbyists with access to a few cloud GPUs should be able get something enjoyable out of this code in a one or a few days. A simple back-of-the-envelope calculation suggests we can speed up execution by 5-10x by lowering the tree search depth, 5-10x by decreasing network depth, 10x by choosing a game with a smaller state space (the “breadth” of a Santorini turn is 32, vs around 35 for chess and 250 for Go), 10-100x by playing fewer games and, this being the key factor, 100-1000x by only requiring enjoyable rather than superhuman performance (there are presumably major diminishing returns to human practice).That gives a total speedup of 125 000 - 100 000 000x.

In case you’re not bothered about sticking rigidly to perfect unsupervised self-play, there are a few ways of boosting training. Especially initial training can be tricky in unsupervised settings given low compute. This might be aided using the history file 1600_no_net.pickle, which contains 5000 triples of (board state, outcome, search probabilities), where outcome is discounted by 0.97^t for each state from the end and the searches were generated with a depth of 1600 and a naïve evaluation function. Also included is toy_problems.py, a few “Santorini joseki” that can be used either to direct initial training, or preferably as test positions for quick qualitative evaluation.

This is still work in progress, and I have not yet trained Alpha Santorini to the level I’d like. However, the current code is able to outperform a naive tree search after a few hours of training, using the tricks just mentioned.
The search for better parameters continues!
