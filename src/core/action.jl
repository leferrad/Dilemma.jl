export
    Action


"""
    Action(choice, space)
    Action(choice; k)

Represents the action of pulling a `Bandit`'s arm that was chosen by a `Policy`.

This is handled directly by a `Policy`, so it should be transparent for the user in the program.

# Fields
- `choice::Int`: Index of arm selected by a `Policy`
- `space::Union{Tuple{Vararg{Symbol}}, Nothing}`: Tuple with symbols that identify 
    the possible arms to select from a given `Bandit`.
- `k::Union{Integer, Nothing}`: to create a space of k choices.
```
"""
mutable struct Action
    choice::Int
    space::Union{Tuple{Vararg{Symbol}}, Nothing}

    function Action(
        choice::Integer,
        space::Union{Tuple{Vararg{Symbol}}, Nothing}
    )
        if choice <= 0
            throw(ArgumentError("Argument 'choice' must be positive"))
        end

        if space !== nothing && (choice > length(space))
            throw(ArgumentError("Argument 'choice' is not in space of 1:$(length(space)) options"))
        end

        new(choice, space)
    end

    function Action(choice::Integer; k::Union{Integer, Nothing}=nothing)
        space = k === nothing ? 
            nothing : 
            Tuple(Symbol("a$i") for i in 1:k)
        return Action(choice, space)
    end

end
