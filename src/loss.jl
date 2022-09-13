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
# function update_∇!(∇¹, ∇², x, y, p, w)
#     @inbounds for j in axes(x, 2)
#         @inbounds for i in axes(x, 1)
#             ∇¹[j] += 2 * x[i, j] * (p[i] - y[i])
#             ∇²[j] += 2 * x[i, j]^2
#         end
#     end
#     return nothing
# end
function update_∇¹!(∇¹, x, y, p, w)
    @inbounds for j in axes(x, 2)
        @inbounds for i in axes(x, 1)
            ∇¹[j] += 2 * x[i, j] * (p[i] - y[i])
        end
    end
    return nothing
end
function update_∇²!(∇², x, y, p, w)
    @inbounds for j in axes(x, 2)
        @inbounds for i in axes(x, 1)
            ∇²[j] += 2 * x[i, j]^2
        end
    end
    return nothing
end


###################################
# logistic
###################################
function update_∇¹!(∇¹, x, y, p, w)
    @inbounds for j in axes(x, 2)
        @inbounds for i in axes(x, 1)
            ∇¹[j] += 2 * x[i, j] * (p[i] - y[i])
        end
    end
    return nothing
end
function update_∇²!(∇², x, y, p, w)
    @inbounds for j in axes(x, 2)
        @inbounds for i in axes(x, 1)
            ∇²[j] += 2 * x[i, j]^2
        end
    end
    return nothing
end


function mse(pred, y)
    return mean((pred .- y) .^ 2)
end
# function mse(pred, y, w)
#     return mean((pred .- y) .^ 2)
# end