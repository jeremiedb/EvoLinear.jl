function fit(config::EvoLinearRegressor, x, y)

    ### init
    w = ones(length(y))

    p = zeros(length(y))
    fill!(p, mean(y))

    m = EvoLearner(zeros(size(x, 2)), zero(Float64))
    
    ∇² = init_∇²(x, w)
    ∇¹ = init_∇¹(x)    
    # fit loop
    update_∇¹!(∇¹, x, y, p, w)
    # update_β!(m, ∇¹, ∇²)
    # update_bias!(m, ∇¹, ∇²)

    return ∇¹, ∇²
    # return nothing
end