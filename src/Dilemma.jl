module Dilemma

include("core/action.jl")
include("core/context.jl")
include("core/reward.jl")
include("core/util.jl")
include("bandit/bandit.jl")
include("bandit/stochastic/base.jl")
include("bandit/stochastic/bernoulli.jl")
include("policy/policy.jl")
include("policy/basic/random.jl")

Dilemma

end # module
