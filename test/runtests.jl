using Dilemma
using Test

@testset "Dilemma" begin
    include("test_utils.jl")
    include("core/action.jl")
    include("core/agent.jl")
    include("core/context.jl")
    include("core/history.jl")
    include("core/reward.jl")
    include("core/util.jl")
    include("bandit/stochastic/base.jl")
    include("bandit/stochastic/bernoulli.jl")
    include("bandit/stochastic/beta.jl")
    include("bandit/stochastic/gaussian.jl")
    include("bandit/stochastic/uniform.jl")
    include("policy/policy.jl")
    include("policy/basic/random.jl")
    include("policy/context_free/e_greedy.jl")
    include("policy/contextual/e_greedy.jl")
end
