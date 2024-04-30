function init_∇¹(x)
    ∇¹ = zeros(size(x, 2))
    return ∇¹
end
function init_∇²(x)
    ∇² = zeros(size(x, 2))
    return ∇²
end

"""
    update_∇!(L, ∇¹, ∇², x, y, p, w)

Update gradients w.r.t each feature. Each feature gradient update is dispatch according to the loss type (`mse`, `logistic`...).
"""
function update_∇!(L, ∇¹, ∇², x, y::A, p::A, w::A) where {A}
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
    @turbo for i in axes(x, 1)
        ∇1 += 2 * (p[i] - y[i]) * x[i, feat] * w[i]
        ∇2 += 2 * x[i, feat]^2 * w[i]
    end
    ∇¹[feat] = ∇1
    ∇²[feat] = ∇2
    return nothing
end
function update_∇_bias!(::Type{MSE}, ∇_bias, x, y, p, w)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * (p[i] - y[i]) * w[i]
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
        ∇1 += (p[i] - y[i]) * x[i, feat] * w[i]
        ∇2 += p[i] * (1 - p[i]) * x[i, feat]^2 * w[i]
    end
    ∇¹[feat] = ∇1
    ∇²[feat] = ∇2
    return nothing
end
function update_∇_bias!(::Type{Logistic}, ∇_bias, x, y, p, w)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += (p[i] - y[i]) * w[i]
        ∇2 += p[i] * (1 - p[i]) * w[i]
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
function update_∇!(::Type{PoissonDev}, ∇¹, ∇², x, y, p, w, feat)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * (p[i] - y[i]) * x[i, feat] * w[i]
        ∇2 += 2 * p[i] * x[i, feat]^2 * w[i]
    end
    ∇¹[feat] = ∇1
    ∇²[feat] = ∇2
    return nothing
end
function update_∇_bias!(::Type{PoissonDev}, ∇_bias, x, y, p, w)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * (p[i] - y[i]) * w[i]
        ∇2 += 2 * p[i] * w[i]
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
function update_∇!(::Type{GammaDev}, ∇¹, ∇², x, y, p, w, feat)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * (1 - y[i] / p[i]) * x[i, feat] * w[i]
        ∇2 += 2 * y[i] / p[i] * x[i, feat]^2 * w[i]
    end
    ∇¹[feat] = ∇1
    ∇²[feat] = ∇2
    return nothing
end
function update_∇_bias!(::Type{GammaDev}, ∇_bias, x, y, p, w)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * (1 - y[i] / p[i]) * w[i]
        ∇2 += 2 * y[i] / p[i] * w[i]
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
function update_∇!(::Type{TweedieDev}, ∇¹, ∇², x, y, p, w, feat)
    rho = eltype(p)(1.5)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * (p[i]^(2 - rho) - y[i] * p[i]^(1 - rho)) * x[i, feat] * w[i]
        ∇2 += 2 * ((2 - rho) * p[i]^(2 - rho) - (1 - rho) * y[i] * p[i]^(1 - rho)) * x[i, feat]^2 * w[i]
    end
    ∇¹[feat] = ∇1
    ∇²[feat] = ∇2
    return nothing
end
function update_∇_bias!(::Type{TweedieDev}, ∇_bias, x, y, p, w)
    rho = eltype(p)(1.5)
    ∇1, ∇2 = zero(eltype(p)), zero(eltype(p))
    @turbo for i in axes(x, 1)
        ∇1 += 2 * (p[i]^(2 - rho) - y[i] * p[i]^(1 - rho)) * w[i]
        ∇2 += 2 * ((2 - rho) * p[i]^(2 - rho) - (1 - rho) * y[i] * p[i]^(1 - rho)) * w[i]
    end
    ∇_bias[1] = ∇1
    ∇_bias[2] = ∇2
    return nothing
end