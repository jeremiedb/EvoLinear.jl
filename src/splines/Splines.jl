module Splines

using ..EvoLinear

import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema

export EvoSplineRegressor

using Statistics
using Distributions: Normal
using Flux
using Flux: DataLoader
using Optimisers
using Flux: update!, @functor
using SparseArrays
using LinearAlgebra
using Random

mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(rng::T) where {T<:Integer} = Random.MersenneTwister(rng)

include("models.jl")
include("structs.jl")
include("fit.jl")

end