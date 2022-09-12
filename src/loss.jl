# linear
function init_∇¹(x)
    ∇¹ = zeros(size(x, 2))
    return ∇¹
end

function update_∇¹!(∇¹, x, y, p, w)
    for j in axes(x, 2)
        for i in axes(x, 1)
            ∇¹[j] += 2 * x[i, j] * (p[i] - y[i])
        end
    end
    return nothing
end

function init_∇²(x, w)
    ∇² = zeros(size(x, 2))
    for j in axes(x, 2)
        for i in axes(x, 1)
            ∇²[j] += 2 * x[i, j]^2
        end
    end
    return ∇²
end