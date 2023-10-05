module Splines

using ..EvoLinear
using ..EvoLinear: sigmoid, logit, mk_rng

using ..EvoLinear.Metrics
using ..EvoLinear.Losses
using ..EvoLinear.CallBacks
import ..EvoLinear.CallBacks: CallBackLinear
import ..EvoLinear: init, fit!, get_loss_type

import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema

using Statistics: mean, std
using StatsBase: quantile
using Distributions: Normal
using Flux
using Flux: DataLoader
using Flux: update!, @functor
using Optimisers
using SparseArrays
using LinearAlgebra

export EvoSplineRegressor, EvoSplineModel

include("models.jl")
include("loss.jl")
include("structs.jl")
include("fit.jl")

end