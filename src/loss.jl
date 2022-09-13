function init_∇¹(x)
    ∇¹ = zeros(size(x, 2))
    return ∇¹
end
function init_∇²(x)
    ∇² = zeros(size(x, 2))
    return ∇²
end


###################################
# linear
###################################
function update_∇!(L::Type{MSE}, ∇¹, ∇², x, y, p, w)
    @inbounds for feat in axes(x, 2)
        @inbounds for i in axes(x, 1)
            ∇¹[feat] += 2 * x[i, feat] * (p[i] - y[i])
            ∇²[feat] += 2 * x[i, feat]^2
        end
    end
    return nothing
end
function update_∇¹!(L::Type{MSE}, ∇¹, x, y, p, w)
    @inbounds for feat in axes(x, 2)
        update_∇¹!(L, ∇¹, x, y, p, w, feat)
    end
    return nothing
end
function update_∇¹!(L::Type{MSE}, ∇¹, x, y, p, w, feat)
    @inbounds for i in axes(x, 1)
        ∇¹[feat] += 2 * x[i, feat] * (p[i] - y[i])
    end
    return nothing
end
function update_∇²!(L::Type{MSE}, ∇², x, y, p, w)
    @inbounds for feat in axes(x, 2)
        update_∇²!(L, ∇², x, y, p, w, feat)
    end
    return nothing
end
function update_∇²!(L::Type{MSE}, ∇², x, y, p, w, feat)
    @inbounds for i in axes(x, 1)
        ∇²[feat] += 2 * x[i, feat]^2
    end
    return nothing
end

###################################
# logistic
###################################
function logit(x::AbstractArray{T,1}) where {T<:AbstractFloat}
    logit.(x)
end
function logit(x::T) where {T<:AbstractFloat}
    log(x / (1 - x))
end

function sigmoid(x::AbstractArray{T,1}) where {T<:AbstractFloat}
    sigmoid.(x)
end
function sigmoid(x::T) where {T<:AbstractFloat}
    1 / (1 + exp(-x))
end

function update_∇¹!(L::Type{Logistic}, ∇¹, x, y, p, w)
    ps = sigmoid(p)
    @inbounds for feat in axes(x, 2)
        update_∇¹!(L, ∇¹, x, y, ps, w, feat)
    end
    return nothing
end
function update_∇¹!(L::Type{Logistic}, ∇¹, x, y, ps, w, feat)
    @inbounds for i in axes(x, 1)
        ∇¹[feat] += x[i, feat] * (ps[i] - y[i])
    end
    return nothing
end
function update_∇²!(L::Type{Logistic}, ∇², x, y, p, w)
    ps = sigmoid(p)
    @inbounds for feat in axes(x, 2)
        update_∇²!(L, ∇², x, y, ps, w, feat)
    end
    return nothing
end
function update_∇²!(L::Type{Logistic}, ∇², x, y, ps, w, feat)
    @inbounds for i in axes(x, 1)
        ∇²[feat] += ps[i] * (1 - ps[i]) * x[i, feat]^2
    end
    return nothing
end


function mse(pred, y)
    return mean((pred .- y) .^ 2)
end
# function mse(pred, y, w)
#     return mean((pred .- y) .^ 2)
# end