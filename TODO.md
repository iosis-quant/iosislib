# graph
- handle caching
- make the system able to deal with data that is too large to fit into memory

# backtester
- improve accuracy of backtester
- add validation metrics
- improve risk modelling and make it more modular

# nodes
- add more tsfns. murex, scipy, are good starting points [1/10]
- create better "categories" of tsfn where shared interfaces are actually abstracted hierarchically

# adapters
- adapters should fetch content adressed material with calver/semver pointers. we basically have some table in s3 that we [x] 

# charting
- make the charts more visually appealing [x]

# testing
- test that errors are found throughout the program with a single pass of the verifier
- test performance more rigorously (different graph complexities, backtest sample sizes etc)