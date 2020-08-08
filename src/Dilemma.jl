module Dilemma

include("core/action.jl")
include("core/context.jl")
include("core/reward.jl")
include("core/util.jl")
include("bandit/bandit.jl")
include("bandit/stochastic/base.jl")
include("bandit/stochastic/bernoulli.jl")
include("bandit/stochastic/beta.jl")
include("policy/policy.jl")
include("policy/basic/random.jl")
include("policy/context_free/e_greedy.jl")
include("policy/contextual/e_greedy.jl")

Dilemma

end # module
