module EvoLinear

using Statistics: mean, std
import Random

import MLJModelInterface as MMI
import MLJModelInterface: fit, update, predict, schema

export EvoLinearRegressor, EvoSplineRegressor

include("utils.jl")
include("metrics.jl")
include("callback.jl")
include("losses.jl")

include("linear/Linear.jl")
using .Linear

include("splines/Splines.jl")
using .Splines

const EvoLinearTypes = Union{EvoLinearRegressor,EvoSplineRegressor}

include("MLJ.jl")

end
