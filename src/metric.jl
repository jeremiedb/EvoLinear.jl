
"""
mse(p, y)

Mean squared error evaluation metric.
"""
function mse(p, y)
    return mean((p .- y) .^ 2)
end

"""
mae(p, y)

Mean absolute error evaluation metric.
"""
function mae(p, y)
    return mean(abs.(p .- y))
end

"""
logloss(p, y)

Logloss evaluation metric.
ylog(p) + (1-y)log(1-p)
"""
function logloss(p, y)
    ϵ = eps(eltype(y)(1e-7))
    return -mean(y .* log.(p .+ ϵ) .+ (1 .- y) .* log.(1 .- p .+ ϵ))
end

"""
poisson(p, y)

Poisson deviance evaluation metric.
𝐷 = 2 * (y * log(y/p) + p - y)
"""
function poisson(p, y)
    ϵ = eps(eltype(p)(1e-7))
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += 2 * (y[i] * log(y[i] / p[i] + ϵ) + p[i] - y[i])
    end
    return metric / length(p)
end
function poisson(p, y, w)
    ϵ = eps(eltype(p)(1e-7))
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += 2 * (y[i] * log(y[i] / p[i] + ϵ) + p[i] - y[i]) * w[i]
    end
    return metric / sum(w)
end

"""
gamma(p, y)

Gamma deviance evaluation metric.
𝐷 = 2 * (log(μ/y) + y/μ - 1)
"""
function gamma(p, y)
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += 2 * (log(p[i] / y[i]) + y[i] / p[i] - 1)
    end
    return metric / length(p)
end
function gamma(p, y, w)
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += 2 * (log(p[i] / y[i]) + y[i] / p[i] - 1) * w[i]
    end
    return metric / sum(w)
end

"""
tweedie(p, y)

Tweedie deviance evaluation metric.
𝐷 = 2 * (y²⁻ʳʰᵒ/(1-rho)(2-rho) - yμ¹⁻ʳʰᵒ/(1-rho) + μ²⁻ʳʰᵒ/(2-rho))
"""
function tweedie(p, y)
    rho = eltype(p)(1.5)
    metric = zero(eltype(p))
    @turbo for i in eachindex(y)
        metric += 2 * (y[i]^(2 - rho) / (1 - rho) / (2 - rho) - y[i] * p[i]^(1 - rho) / (1 - rho) + p[i]^(2 - rho) / (2 - rho))
    end
    return metric / length(p)
end
function tweedie(p, y, w)
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
    :poisson => poisson,
    :gamma => gamma,
    :tweedie => tweedie
)