module Losses

export Loss, MSE, Logistic, Poisson, Gamma, Tweedie, loss_types

abstract type Loss end
struct MSE <: Loss end
struct Logistic <: Loss end
struct Poisson <: Loss end
struct Gamma <: Loss end
struct Tweedie <: Loss end

const loss_types = Dict(
    :mse => MSE,
    :logistic => Logistic,
    :poisson => Poisson,
    :gamma => Gamma,
    :tweedie => Tweedie,
)

end
