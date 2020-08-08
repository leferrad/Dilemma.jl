using Dilemma
using Test

@testset "Dilemma" begin
    include("test_utils.jl")
    include("core/action.jl")
    include("core/context.jl")
    include("core/reward.jl")
    include("core/util.jl")
    include("bandit/stochastic/base.jl")
    include("bandit/stochastic/bernoulli.jl")
    include("policy/policy.jl")
    include("policy/basic/random.jl")
end
