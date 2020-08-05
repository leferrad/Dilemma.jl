using Distributions
using Random

export
    Context,
    ContextFunction,
    RandomContextFunction

"""
    Context(args)
    
Represents the context of a `Bandit` that can be used by a `Policy` for the arm selection.

This is handled directly by the program, so it should be transparent for the user in the program.

    Context(x::NamedTuple)
    Context(x::Dict{Symbol,T}) where {T<:Real}
    Context(x::Vector{T}) where {T<:Real}
    Context(x::Tuple{Vararg{T}}) where {T<:Real}

Build a context vector based on `x` that could be an iterable with context values or even have the keys 
    for each of these values. They fields are identified with Symbols, and the values could be of Any type.

    Context()

No context information defined (all fields having `nothing`).

If you want to get the context values (without keys), run `collect()` over the Context instance.

# Fields
- `x::Union{NamedTuple, Nothing}`: having keys and values that define a context instance
- `d::Union{Integer, Nothing}`: number of dimensions of context
- `keys::Union{Tuple{Vararg{Symbol}}, Nothing} `: names of fields that identify the context instance
- `values::Union{Tuple{Vararg{Any}}, Nothing}`: context values for the given fields

# Examples
```julia-repl
julia> using Dilemma
julia> context = Context((a=1, b=2, c=3))
Context((a = 1, b = 2, c = 3), 3, (:a, :b, :c), (1, 2, 3))
julia> collect(context)
3-element Array{Int64,1}:
 1
 2
 3
julia> context = Context([1, 2, 3])
Context((x1 = 1, x2 = 2, x3 = 3), 3, (:x1, :x2, :x3), (1, 2, 3))
julia> collect(context)
3-element Array{Int64,1}:
 1
 2
 3
```
"""
mutable struct Context
    x::Union{NamedTuple, Nothing}           
    d::Union{Integer, Nothing}             
    keys::Union{Tuple{Vararg{Symbol}}, Nothing} 
    values::Union{Tuple{Vararg{Any}}, Nothing}   #TODO: numerical values only?
end

Context() = Context(nothing, nothing, nothing, nothing)
Context(x::NamedTuple) = Context(x, length(x), keys(x), values(x))
Context(x::Dict{Symbol,T}) where {T<:Any} = Context((;x...))
Context(x::Vector{T}) where {T<:Any} = Context((;[(Symbol("x$i"), v) for (i,v) in enumerate(x)]...))
Context(x::Tuple{Vararg{Any}}) = Context(collect(x))

Base.collect(context::Context) = context.values !== nothing ? collect(context.values) : nothing 



"""
    ContextFunction

Abstraction to define functions that return `Context` instances. These functions are of the kind:
    
    func(t::Integer; kwargs...)

where `t` is the time of the simulation, and some kwargs are accepted in the call.

To check an implementation, see `RandomContextFunction`
"""
abstract type ContextFunction <: Function end
(c::ContextFunction)(t::Integer; kwargs...) = throw(ErrorException("Not implemented!"))


"""
    RandomContextFunction(D::NamedTuple; seed=123)
    RandomContextFunction(D::Dict{Symbol,T}; seed=123) where {T<:Distribution}
    RandomContextFunction(D::Vector{T}; seed=123) where {T<:Distribution}
    RandomContextFunction(D::Tuple{Vararg{Distribution}}; seed=123)
    RandomContextFunction(d::Integer, dist::T=Uniform(); seed=123) where {T<:Distribution}

Set a function that returns `Context` instances following a set of `Distribution` instances 
    to generate `Context` instances with random context values. Then, you can use it as a function like:

    func(t::Integer; kwargs...)

where `t` is the time of the simulation (not used here), and some kwargs are accepted in the call.

# Fields
- `D::Tuple{Vararg{Distribution}}`: Distribution instances to sample from the context values
- `keys::Tuple{Vararg{Symbol}}`: names of fields for resulting context instances
- `keys::Union{Tuple{Vararg{Symbol}}, Nothing} `: names of fields that identify the context instance
- `values::Union{Tuple{Vararg{Any}}, Nothing}`: context values for the given fields

# Examples
```julia-repl
julia> using Dilemma
julia> using Distributions
julia> ctx_func = RandomContextFunction((a=Uniform(), b=Normal(), c=Bernoulli()), seed=123)
(::RandomContextFunction) (generic function with 2 methods)
julia> ctx_func(1)
Context((a = 0.7684476751965699, b = 2.04817970778924, c = false), 3, (:a, :b, :c), (0.7684476751965699, 2.04817970778924, false))
"""
mutable struct RandomContextFunction <: ContextFunction
    D::Tuple{Vararg{Distribution}}
    keys::Tuple{Vararg{Symbol}}
    d::Int
    rng::MersenneTwister
end

RandomContextFunction(D::NamedTuple; seed=123) = RandomContextFunction(values(D), keys(D), length(D), MersenneTwister(seed))
RandomContextFunction(D::Dict{Symbol,T}; seed=123) where {T<:Distribution} = RandomContextFunction((;D...), seed=seed)
RandomContextFunction(D::Vector{T}; seed=123) where {T<:Distribution} = RandomContextFunction((;[(Symbol("x$i"), v) for (i,v) in enumerate(D)]...), seed=seed)
RandomContextFunction(D::Tuple{Vararg{Distribution}}; seed=123) = RandomContextFunction(collect(D), seed=seed)
RandomContextFunction(d::Integer, dist::T=Uniform(); seed=123) where {T<:Distribution} = RandomContextFunction([dist for _ in 1:d], seed=seed)

(c::RandomContextFunction)(t::Integer; kwargs...) = Context((;[(k, rand(c.rng, dist)) for (k, dist) in zip(c.keys, c.D)]...))
