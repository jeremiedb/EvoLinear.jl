module Splines

using ..EvoLinear

export EvoSplineRegressor

import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema

using Statistics
using Distributions
using Flux
using Optimisers
using Flux: update!, @functor
using MLUtils
using SparseArrays
using LinearAlgebra
using Random

mk_rng(rng::Random.AbstractRNG) = rng
mk_rng(rng::T) where {T<:Integer} = Random.MersenneTwister(rng)

include("models.jl")
include("structs.jl")
include("fit.jl")

end