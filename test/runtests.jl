using Dilemma
using Test

@testset "Dilemma" begin
    include("test_utils.jl")
    include("core/action.jl")
    include("core/context.jl")
    include("core/util.jl")
end
