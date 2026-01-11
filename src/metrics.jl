module Metrics

using LoopVectorization

export metric_dict, is_maximise

"""
    mse(p, y)
    mse(p, y, w)

Mean squared error evaluation metric.

# Arguments

- `p`: predicted value.
- `y`: observed target variable.
- `w`: vector of weights.
"""
function mse(p, y)
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += (y[i] - p[i])^2
    end
    return metric / length(p)
end
function mse(p, y, w)
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += (y[i] - p[i])^2 * w[i]
    end
    return metric / sum(w)
end

"""
    rmse(p, y)
    rmse(p, y, w)

Root-Mean squared error evaluation metric.

# Arguments

- `p`: predicted value.
- `y`: observed target variable.
- `w`: vector of weights
"""
rmse(p, y) = sqrt(mse(p, y))
rmse(p, y, w) = sqrt(mse(p, y, w))


"""
    mae(p, y)
    mae(p, y, w)

Mean absolute error evaluation metric.

# Arguments

- `p`: predicted value.
- `y`: observed target variable.
- `w`: vector of weights.
"""
function mae(p, y)
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += abs(y[i] - p[i])
    end
    return metric / length(p)
end
function mae(p, y, w)
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += abs(y[i] - p[i]) * w[i]
    end
    return metric / sum(w)
end

"""
    logloss(p, y)
    logloss(p, y, w)

Logloss evaluation metric.
ylog(p) + (1-y)log(1-p)

# Arguments

- `p`: predicted value. Assumes that p is on a projected basis (ie. in the `[0-1]` range).
- `y`: observed target variable.
- `w`: vector of weights.
"""
function logloss(p, y)
    ϵ = eps(eltype(y)(1e-7))
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += -(y[i] * log(p[i] + ϵ) + (1 - y[i]) * log(1 - p[i] + ϵ))
    end
    return metric / length(p)
end
function logloss(p, y, w)
    ϵ = eps(eltype(y)(1e-7))
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += -(y[i] * log(p[i] + ϵ) + (1 - y[i]) * log(1 - p[i] + ϵ)) * w[i]
    end
    return metric / sum(w)
end

"""
    poisson_deviance(p, y)
    poisson_deviance(p, y, w)

Poisson deviance evaluation metric.
`𝐷 = 2 * (y * log(y/p) + p - y)`

# Arguments

- `p`: predicted value. Assumes that p is on a projected basis (ie. in the `[0-Inf]` range).
- `y`: observed target variable.
- `w`: vector of weights.
"""
function poisson_deviance(p, y)
    ϵ = eps(eltype(p)(1e-7))
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += 2 * (y[i] * log(y[i] / p[i] + ϵ) + p[i] - y[i])
    end
    return metric / length(p)
end
function poisson_deviance(p, y, w)
    ϵ = eps(eltype(p)(1e-7))
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += 2 * (y[i] * log(y[i] / p[i] + ϵ) + p[i] - y[i]) * w[i]
    end
    return metric / sum(w)
end

"""
    gamma_deviance(p, y)
    gamma_deviance(p, y, w)

Gamma deviance evaluation metric.
`𝐷 = 2 * (log(μ/y) + y/μ - 1)`

# Arguments

- `p`: predicted value. Assumes that p is on a projected basis (ie. in the `[0-Inf]` range).
- `y`: observed target variable.
- `w`: vector of weights.
"""
function gamma_deviance(p, y)
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += 2 * (log(p[i] / y[i]) + y[i] / p[i] - 1)
    end
    return metric / length(p)
end
function gamma_deviance(p, y, w)
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += 2 * (log(p[i] / y[i]) + y[i] / p[i] - 1) * w[i]
    end
    return metric / sum(w)
end

"""
    tweedie_deviance(p, y)
    tweedie_deviance(p, y, w)

Tweedie deviance evaluation metric. Fixed rho (ρ) of 1.5.
𝐷 = 2 * (y²⁻ʳʰᵒ/(1-rho)(2-rho) - yμ¹⁻ʳʰᵒ/(1-rho) + μ²⁻ʳʰᵒ/(2-rho))

# Arguments

- `p`: predicted value. Assumes that p is on a projected basis (ie. in the `[0-Inf]` range).
- `y`: observed target variable.
- `w`: vector of weights.
"""
function tweedie_deviance(p, y)
    rho = eltype(p)(1.5)
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += 2 * (y[i]^(2 - rho) / (1 - rho) / (2 - rho) - y[i] * p[i]^(1 - rho) / (1 - rho) + p[i]^(2 - rho) / (2 - rho))
    end
    return metric / length(p)
end
function tweedie_deviance(p, y, w)
    rho = eltype(p)(1.5)
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += 2 * (y[i]^(2 - rho) / (1 - rho) / (2 - rho) - y[i] * p[i]^(1 - rho) / (1 - rho) + p[i]^(2 - rho) / (2 - rho)) * w[i]
    end
    return metric / sum(w)
end

const metric_dict = Dict(
    :mse => mse,
    :rmse => rmse,
    :mae => mae,
    :logloss => logloss,
    :poisson => poisson_deviance,
    :gamma => gamma_deviance,
    :tweedie => tweedie_deviance
)

is_maximise(::typeof(mse)) = false
is_maximise(::typeof(rmse)) = false
is_maximise(::typeof(mae)) = false
is_maximise(::typeof(logloss)) = false
is_maximise(::typeof(poisson_deviance)) = false
is_maximise(::typeof(gamma_deviance)) = false
is_maximise(::typeof(tweedie_deviance)) = false

end # module
