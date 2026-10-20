using Random
include(joinpath(@__DIR__, "..", "vendor", "Resamplings.jl", "src", "Resamplings.jl"))
using .Resamplings
const cache = Dict()
for line in eachline(stdin)
    method, seed, weight_string = split(line, '\t')
    w = parse.(Float64, split(weight_string, ','))
    n = length(w)
    res = get!(cache, (method, n)) do
        Resampling{Symbol(method)}(n)
    end
    ind = zeros(Int, n)
    Random.seed!(parse(Int,seed))
    resample!(res, ind, w, MersenneTwister(parse(Int, seed)))
    println(join(ind, ',')); flush(stdout)
end
