Dilemma.jl
==========

<img style="display: inline;" src="docs/src/assets/logo.png" width="300"/>

*A Julia package to develop and evaluate context-free and contextual Multi-Armed Bandit policies*

[![Build Status](https://travis-ci.org/leferrad/OCReract.jl.svg?branch=master)](https://travis-ci.org/leferrad/Dilemma.jl)
[![Documentation](https://img.shields.io/badge/docs-dev-blue.svg)](https://leferrad.github.io/Dilemma.jl/dev)
[![Coverage Status](https://codecov.io/gh/leferrad/Dilemma.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/leferrad/Dilemma.jl)
[![Join the chat at https://gitter.im/Dilemma.jl](https://badges.gitter.im/Dilemma.jl.svg)](https://gitter.im/Dilemma-jl?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)

## Overview

Dilemma.jl aims to provide a framework to


This package was developed with these ...:


- Usage in real-life scenarios ??

- Good documentation for users that want to learn these techniques to introduce.

Simple and extensible: provide components and interfaces to make simple to use available implementations and also develop new ones for any application of multi-armed bandits.

Easy experimentation: Make it easy for new users to run benchmark experiments, compare different algorithms, evaluate and diagnose agents.

- **Batteries included!** The most popular algorithms are already implemented, ready for usage.


Use binder for examples!! https://mybinder.org/v2/gh/ablaom/MachineLearningInJulia2020/master?filepath=tutorials.ipynb


**Important links**:
  * [Documentation](https://leferrad.github.io/Dilemma.jl/dev)
  * [Examples](https://github.com/leferrad/Dilemma.jl/tree/master/examples)
  * Changes: see [NEWS.md](https://github.com/leferrad/Dilemma.jl/tree/master/NEWS.md)


Check out the [quick start]() page for more details!

> NOTE: Package is still under development, so it is not ready for production purposes



## Implementations

List policies and bandits implemented


| Policy group | Algorithms |
| ------- | ------ |
| Basic  | OraclePolicy, RandomPolicy |
| Context-free | EpsilonGreedyPolicy, Exp3Policy, SoftmaxPolicy |
| Contextual  | EpsilonGreedyPolicy, Linear Thompson Sampling
, LinUCB  |

## Installation

From the Julia REPL, type `]` to enter the Pkg REPL mode and run:
```julia-repl
pkg> add https://github.com/leferrad/Dilemma.jl
```

## Usage

This is a simple example of usage. For more details check the [Documentation](https://leferrad.github.io/Dilemma.jl/dev).

```julia
using Dilemma

# Set a simple stochastic bandit
k, p = 3, 0.7
bandit = BernoulliBandit(k, p)

# Set agents with several policies for simulations
agents = [
    Agent(EpsilonGreedyPolicy(0.1), bandit, "ϵ-greedy Bernoulli"),
    Agent(ContextuaLinTSPolicy(0.1), bandit, "ctx LinTS Bernoulli"),
    Agent(ContextuaLinUCBPolicy(0.1), bandit, "ctx LinUCB Bernoulli"),
]

# Create simulator
horizon = 100  
repetitions = 10
simulator = Simulator(agents, horizon, repetitions)

# Run simulation
history = run_simulation!(simulator)

# Plot results
plot_simulations(history, type="cumulative", regret=false, rate=true)
```

show resulting plot...

## Testing

In a Julia session, run `Pkg.test("Dilemma", coverage=true)`, or just run `julia --code-coverage=all --inline=no test/runtests.jl`.

## Contributing

Please report any issues via the Github [issue tracker](https://github.com/leferrad/Dilemma.jl/issues). All types of issues are welcome and encouraged; this includes bug reports, documentation typos, feature requests, etc.


Give a star!

## License

Dilemma is released under the [MIT License](LICENSE).
