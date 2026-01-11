"""
    predict(m::EvoLinearModel, data; proj::Bool=true)

Predictions from an EvoLinear model.
"""
function predict(m::EvoLinearModel{L}, data; proj::Bool=true) where {L}
    p = m(data; proj)
    return p
end

function (m::EvoLinearModel{L})(x::AbstractMatrix; proj::Bool=true) where {L}
    p = x * m.coef .+ m.bias
    proj ? proj!(L, p) : nothing
    return p
end
function (m::EvoLinearModel{L})(p::AbstractVector, x::AbstractMatrix; proj::Bool=true) where {L}
    p .= x * m.coef .+ m.bias
    proj ? proj!(L, p) : nothing
    return nothing
end
function (m::EvoLinearModel{L})(data; proj::Bool=true) where {L}

    Tables.istable(data) || error("data must be Table compatible")

    T = Float32
    feature_names = m.info[:feature_names]
    nobs = Tables.DataAPI.nrow(data)
    nfeats = length(feature_names)

    x = zeros(T, nobs, nfeats)
    @threads for j in axes(x, 2)
        @views x[:, j] .= Tables.getcolumn(data, feature_names[j])
    end

    p = x * m.coef .+ m.bias
    proj ? proj!(L, p) : nothing
    return p
end


"""
    proj!(::L, p)

Performs a reprojection from raw linear basis to loss-specific actual prediction.
"""
function proj!(::L, p) where {L<:Type{MSE}}
    return nothing
end
function proj!(::L, p) where {L<:Type{LogLoss}}
    p .= sigmoid.(p)
    return nothing
end
function proj!(::L, p) where {L<:Union{Type{Poisson},Type{Gamma},Type{Tweedie}}}
    p .= exp.(p)
    return nothing
end
