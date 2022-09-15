function init_∇¹(x)
    ∇¹ = zeros(size(x, 2))
    return ∇¹
end
function init_∇²(x)
    ∇² = zeros(size(x, 2))
    return ∇²
end

##################################
# Utilities
##################################
function logit(x::AbstractArray{T}) where {T<:AbstractFloat}
    logit.(x)
end
@inline function logit(x::T) where {T<:AbstractFloat}
    @fastmath log(x / (1 - x))
end

function sigmoid(x::AbstractArray{T}) where {T<:AbstractFloat}
    sigmoid.(x)
end
@inline function sigmoid(x::AbstractFloat)
    t = @fastmath exp(-abs(x))
    y = ifelse(x ≥ 0, inv(1 + t), t / (1 + t))
    ifelse(x > 40, one(y), ifelse(x < -80, zero(y), y))
end

function update_∇!(L, ∇¹, ∇², x, y, p, w)
    @threads for feat in axes(x, 2)
        update_∇!(L, ∇¹, ∇², x, y, p, w, feat)
    end
    return nothing
end

###################################
# linear
###################################
function update_∇!(::Type{MSE}, ∇¹, ∇², x, y, p, w, feat)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @inbounds for i in axes(x, 1)
        ∇1 += 2 * w[i] * (p[i] - y[i]) * x[i, feat]
        ∇2 += 2 * w[i] * x[i, feat]^2
    end
    ∇¹[feat] = ∇1
    ∇²[feat] = ∇2
    return nothing
end
function update_∇_bias!(::Type{MSE}, ∇_bias, x, y, p, w)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @inbounds for i in axes(x, 1)
        ∇1 += 2 * w[i] * (p[i] - y[i])
        ∇2 += 2 * w[i]
    end
    ∇_bias[1] = ∇1
    ∇_bias[2] = ∇2
    return nothing
end

###################################
# logistic
###################################
function update_∇!(::Type{Logistic}, ∇¹, ∇², x, y, p, w, feat)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += w[i] * (p[i] - y[i]) * x[i, feat]
        ∇2 += w[i] * p[i] * (1 - p[i]) * x[i, feat]^2
    end
    ∇¹[feat] = ∇1
    ∇²[feat] = ∇2
    return nothing
end
function update_∇_bias!(::Type{Logistic}, ∇_bias, x, y, p, w)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += w[i] * (p[i] - y[i])
        ∇2 += w[i] * p[i] * (1 - p[i])
    end
    ∇_bias[1] = ∇1
    ∇_bias[2] = ∇2
    return nothing
end

###################################
# Poisson
# Deviance = 2 * (y * log(y/μ) + μ - y)
# https://www.casact.org/sites/default/files/old/rpm_2013_handouts_paper_1497_handout_795_0.pdf
# The prediction p is assumed to be on the projected basis (exp(pred_linear))
# Derivative is w.r.t to β on the linear basis
###################################
function update_∇!(::Type{Poisson}, ∇¹, ∇², x, y, p, w, feat)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * w[i] * (p[i] - y[i]) * x[i, feat]
        ∇2 += 2 * w[i] * p[i] * x[i, feat]^2
    end
    ∇¹[feat] = ∇1
    ∇²[feat] = ∇2
    return nothing
end
function update_∇_bias!(::Type{Poisson}, ∇_bias, x, y, p, w)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * w[i] * (p[i] - y[i])
        ∇2 += 2 * w[i] * p[i]
    end
    ∇_bias[1] = ∇1
    ∇_bias[2] = ∇2
    return nothing
end

###################################
# Gamma
# Deviance = 2 * (log(μ/y) + y/μ - 1)
# https://www.casact.org/sites/default/files/old/rpm_2013_handouts_paper_1497_handout_795_0.pdf
# The prediction p is assumed to be on the projected basis (exp(pred_linear))
# Derivative is w.r.t to β on the linear basis
###################################
function update_∇!(::Type{Gamma}, ∇¹, ∇², x, y, p, w, feat)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * w[i] * (1 - y[i] / p[i]) * x[i, feat]
        ∇2 += 2 * w[i] * y[i] / p[i] * x[i, feat]^2
    end
    ∇¹[feat] = ∇1
    ∇²[feat] = ∇2
    return nothing
end
function update_∇_bias!(::Type{Gamma}, ∇_bias, x, y, p, w)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * w[i] * (1 - y[i] / p[i])
        ∇2 += 2 * w[i] * y[i] / p[i]
    end
    ∇_bias[1] = ∇1
    ∇_bias[2] = ∇2
    return nothing
end

###################################
# Tweedie
# Deviance = 2 * (y²⁻ᵖ/(1-p)(2-p) - yμ¹⁻ᵖ/(1-p) + μ²⁻ᵖ/(2-p))
# https://www.casact.org/sites/default/files/old/rpm_2013_handouts_paper_1497_handout_795_0.pdf
# The prediction p is assumed to be on the projected basis (exp(pred_linear))
# Derivative is w.r.t to β on the linear basis
###################################
function update_∇!(::Type{Tweedie}, ∇¹, ∇², x, y, p, w, feat)
    rho = eltype(p)(1.5)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * w[i] * (p[i]^(2 - rho) - y[i] * p[i]^(1 - rho)) * x[i, feat]
        ∇2 += 2 * w[i] * ((2 - rho) * p[i]^(2 - rho) - (1 - rho) * y[i] * p[i]^(1 - rho)) * x[i, feat]^2
    end
    ∇¹[feat] = ∇1
    ∇²[feat] = ∇2
    return nothing
end
function update_∇_bias!(::Type{Tweedie}, ∇_bias, x, y, p, w)
    rho = eltype(p)(1.5)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * w[i] * (p[i]^(2 - rho) - y[i] * p[i]^(1 - rho))
        ∇2 += 2 * w[i] * ((2 - rho) * p[i]^(2 - rho) - (1 - rho) * y[i] * p[i]^(1 - rho))
    end
    ∇_bias[1] = ∇1
    ∇_bias[2] = ∇2
    return nothing
end


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
# function poisson(p, y)
#     ϵ = eps(eltype(p)(1e-7))
#     return mean(2 .* (y .* log.(y ./ p .+ ϵ) .+ p .- y))
# end
function poisson(p, y)
    ϵ = eps(eltype(p)(1e-7))
    metric = zero(eltype(p))
    @tturbo for i in eachindex(y)
        metric += 2 * (y[i] * log(y[i] / p[i] + ϵ) + p[i] - y[i])
    end
    return metric
end

"""
    gamma(p, y)

Gamma deviance evaluation metric.
𝐷 = 2 * (log(μ/y) + y/μ - 1)
"""
function gamma(p, y)
    metric = zero(eltype(p))
    @tturbo for i in eachindex(y)
        metric += 2 * (log(p[i] / y[i]) + y[i] / p[i] - 1)
    end
    return metric
end

"""
    tweedie(p, y)

Tweedie deviance evaluation metric.
𝐷 = 2 * (y²⁻ʳʰᵒ/(1-rho)(2-rho) - yμ¹⁻ʳʰᵒ/(1-rho) + μ²⁻ʳʰᵒ/(2-rho))
"""
function tweedie(p, y)
    rho = eltype(p)(1.5)
    metric = zero(eltype(p))
    @tturbo for i in eachindex(y)
        metric += 2 * (y[i]^(2 - rho) / (1 - rho) / (2 - rho) - y[i] * p[i]^(1 - rho) / (1 - rho) + p[i]^(2 - rho) / (2 - rho))
    end
    return metric
end

const metric_dict = Dict(
    :mse => mse,
    :mae => mae,
    :logloss => logloss,
    :poisson => poisson,
    :gamma => gamma,
    :tweedie => tweedie
)