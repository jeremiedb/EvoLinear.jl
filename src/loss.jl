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
    @inbounds for i in axes(x, 1)
        ∇¹[feat] += 2 * x[i, feat] * (p[i] - y[i])
        ∇²[feat] += 2 * x[i, feat]^2
    end
    return nothing
end
function update_∇_bias!(::Type{MSE}, ∇_bias, x, y, p, w)
    @inbounds for i in axes(x, 1)
        ∇_bias[1] += 2 * (p[i] - y[i])
        ∇_bias[2] += 2
    end
    return nothing
end

###################################
# logistic
###################################
function update_∇!(::Type{Logistic}, ∇¹, ∇², x, y, p, w, feat)
    # p = sigmoid(p)
    @inbounds for i in axes(x, 1)
        ∇¹[feat] += x[i, feat] * (p[i] - y[i])
        ∇²[feat] += p[i] * (1 - p[i]) * x[i, feat]^2
    end
    return nothing
end
function update_∇_bias!(::Type{Logistic}, ∇_bias, x, y, p, w)
    # p = sigmoid(p)
    @inbounds for i in axes(x, 1)
        ∇_bias[1] += (p[i] - y[i])
        ∇_bias[2] += p[i] * (1 - p[i])
    end
    return nothing
end


function mse(pred, y)
    return mean((pred .- y) .^ 2)
end
function mae(pred, y)
    return mean(abs.(pred .- y))
end
function logloss(pred, y)
    return -mean(y .* log.(pred) .+ (1 .- y) .* log.(1 .- pred))
end