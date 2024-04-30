module Losses

export Loss, MSE, Logistic, PoissonDev, GammaDev, TweedieDev, loss_types

abstract type Loss end
struct MSE <: Loss end
struct Logistic <: Loss end
struct PoissonDev <: Loss end
struct GammaDev <: Loss end
struct TweedieDev <: Loss end

const loss_types = Dict(
    :mse => MSE,
    :logistic => Logistic,
    :poisson_deviance => PoissonDev,
    :gamma_deviance => GammaDev,
    :tweedie_deviance => TweedieDev,
)

end
