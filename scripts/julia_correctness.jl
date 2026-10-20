# Run upstream API, input-validation, and allocation checks. The upstream full
# statistical stress tests use 50 million repetitions per configuration; the
# common benchmark suite supplies its own seeded statistical experiments.
using Resamplings, Test
cd(joinpath(@__DIR__, "..", "vendor", "Resamplings.jl", "test"))
source = read("test-resample.jl", String)
source = replace(source, "include(\"helpers.jl\");" => "include(" * repr(joinpath(pwd(), "helpers.jl")) * ");")
include_string(Main, first(split(source, "println(\"Running symmetry tests")), joinpath(pwd(), "test-resample.jl"))
source = read("test-conditional-resample.jl", String)
source = replace(source, "include(\"helpers.jl\");" => "include(" * repr(joinpath(pwd(), "helpers.jl")) * ");")
# Only execute the testsets before the large statistical symmetry section.
source = first(split(source, "## Symmetricity tests for conditional resamplings."))
include_string(Main, source, joinpath(pwd(), "test-conditional-resample.jl"))
