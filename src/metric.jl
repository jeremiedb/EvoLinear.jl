
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
    :mae => mae,
    :logloss => logloss,
    :poisson_deviance => poisson_deviance,
    :gamma_deviance => gamma_deviance,
    :tweedie_deviance => tweedie_deviance
)

is_maximise(::typeof(EvoLinear.mse)) = false
is_maximise(::typeof(EvoLinear.mae)) = false
is_maximise(::typeof(EvoLinear.logloss)) = false
is_maximise(::typeof(EvoLinear.poisson_deviance)) = false
is_maximise(::typeof(EvoLinear.gamma_deviance)) = false
is_maximise(::typeof(EvoLinear.tweedie_deviance)) = false


struct CallBackLinear{F,M,V}
    feval::F
    x::M
    p::V
    y::V
    w::V
end
function (cb::CallBackLinear)(logger, iter, m)
    m(cb.p, cb.x; proj = true)
    metric = cb.feval(cb.p, cb.y, cb.w)
    update_logger!(logger, iter, metric)
    return logger[:early_stopping_rounds] <= logger[:iter_since_best]
end

function CallBackLinear(; metric, x_eval, y_eval, w_eval = nothing, T = Float32)
    feval = metric_dict[metric]
    x = convert(Matrix{T}, x_eval)
    p = zeros(T, length(y_eval))
    y = convert(Vector{T}, y_eval)
    w = isnothing(w_eval) ? ones(T, size(y)) : convert(Vector{T}, w_eval)
    return CallBackLinear(feval, x, p, y, w)
end

function init_logger(; T, metric, maximise, early_stopping_rounds)
    logger = Dict(
        :name => String(metric),
        :maximise => maximise,
        :early_stopping_rounds => early_stopping_rounds,
        :niter => 0,
        :iter => Int[],
        :metrics => T[],
        :iter_since_best => 0,
        :best_iter => 0,
        :best_metric => 0.0,
    )
    return logger
end

function update_logger!(logger, iter, metric)
    logger[:niter] = iter
    push!(logger[:iter], iter)
    push!(logger[:metrics], metric)
    if iter > 0
        if (logger[:maximise] && metric > logger[:best_metric]) ||
           (!logger[:maximise] && metric < logger[:best_metric])
            logger[:best_metric] = metric
            logger[:best_iter] = iter
            logger[:iter_since_best] = 0
        else
            logger[:iter_since_best] += logger[:iter][end] - logger[:iter][end-1]
        end
    end
end