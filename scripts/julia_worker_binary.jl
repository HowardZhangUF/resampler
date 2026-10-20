using Random
include(joinpath(@__DIR__, "..", "vendor", "Resamplings.jl", "src", "Resamplings.jl"))
using .Resamplings
const cache = Dict()
const methods = (:multinomial, :systematic, :stratified, :residual, :ssp, :killing)
while !eof(stdin)
    method = methods[Int(read(stdin, UInt32))+1]
    n = Int(read(stdin, UInt32))
    seed = read(stdin, Int64)
    w = Vector{Float64}(undef,n)
    read!(stdin,w)
    res = get!(cache,(method,n)) do
        Resampling{method}(n)
    end
    ind = zeros(Int64,n)
    # Upstream multinomial uses one global randexp(T) in addition to rng draws.
    Random.seed!(seed)
    resample!(res,ind,w,MersenneTwister(seed))
    write(stdout,ind);flush(stdout)
end
