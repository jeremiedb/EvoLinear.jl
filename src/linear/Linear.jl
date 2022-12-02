module Linear

using ..EvoLinear
using ..EvoLinear: sigmoid, logit, mk_rng

using ..EvoLinear.Metrics
using ..EvoLinear.Losses
using ..EvoLinear.CallBacks
import ..EvoLinear.CallBacks: CallBackLinear

using Base.Threads: @threads
# using StatsBase
using Statistics: mean, std
using LoopVectorization

import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema

export EvoLinearRegressor, EvoLinearModel, init, fit!, get_loss_type

include("structs.jl")
include("loss.jl")
include("predict.jl")
include("fit.jl")

end