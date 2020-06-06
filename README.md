# lmnn
A fast Python implementation of the Large Margin Nearest Neighbor (LMNN) algorithm and it's next door neighbor the Large Margin Component Analysis (LMCA)). The implementation uses a dictionary to cache heavily used matrix products speeding up computations significantly. In addition, there's a number of implemented regularization functions acting on the transform matrix L such as the l1/l2 norm.

## Todo
* Add unittest files
* Refactor.
* Pull matrix product caching out to an interface file.
