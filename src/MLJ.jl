function MMI.fit(model::EvoLinearTypes, verbosity::Int, A, y)
    fitresult, cache = init(model; x=A.matrix, y)

    for i in 1:model.nrounds
        fit!(fitresult, cache, model)
    end
    report = (coef=fitresult.coef, bias=fitresult.bias, names=A.names)
    return fitresult, cache, report
end

function okay_to_continue(model, fitresult, cache)
    model.nrounds - cache[:logger][:nrounds] >= 0 &&
        get_loss_type(fitresult) <: loss_types[model.loss]
end

# Generate names to be used by feature_importances in the report
MMI.reformat(::EvoLinearTypes, X, y) = ((matrix=MMI.matrix(X), names=[name for name ∈ schema(X).names]), y)
MMI.reformat(::EvoLinearTypes, X) = ((matrix=MMI.matrix(X), names=[name for name ∈ schema(X).names]),)
MMI.reformat(::EvoLinearTypes, X::AbstractMatrix, y) = ((matrix=X, names=["feat_$i" for i = 1:size(X, 2)]), y)
MMI.reformat(::EvoLinearTypes, X::AbstractMatrix) = ((matrix=X, names=["feat_$i" for i = 1:size(X, 2)]),)
MMI.selectrows(::EvoLinearTypes, I, A, y) = ((matrix=view(A.matrix, I, :), names=A.names), view(y, I))
MMI.selectrows(::EvoLinearTypes, I, A) = ((matrix=view(A.matrix, I, :), names=A.names))

# For EarlyStopping.jl supportm
MMI.iteration_parameter(::Type{<:EvoLinearTypes}) = :nrounds

function MMI.update(model::EvoLinearTypes, verbosity::Integer, fitresult, cache, A, y)

    if okay_to_continue(model, fitresult, cache)
        nrounds_old = cache[:logger][:nrounds]
        cache = init_cache(model; x=A.matrix, y)
        cache[:logger][:nrounds] = nrounds_old
        while cache[:logger][:nrounds] < model.nrounds
            fit!(fitresult, cache, model)
        end
        report = (coef=fitresult.coef, bias=fitresult.bias, names=A.names)
    else
        fitresult, cache, report = fit(model, verbosity, A, y)
    end

    return fitresult, cache, report
end

function predict(::EvoLinearTypes, fitresult, A)
    pred = fitresult(A.matrix)
    return pred
end

# Metadata
const EvoLinearRegressor_desc = "Regression models for various taks: mse, logistic, poisson, gamma, tweedie."

MMI.metadata_pkg.((EvoLinearRegressor),
    name="EvoLinear",
    uuid="ab853011-1780-437f-b4b5-5de6f4777246",
    url="https://github.com/jeremiedb/EvoLinear.jl",
    julia=true,
    license="MIT",
    is_wrapper=false)

MMI.metadata_model(EvoLinearRegressor,
    input_scitype=Union{MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor),AbstractMatrix{MMI.Continuous}},
    target_scitype=AbstractVector{<:MMI.Continuous},
    weights=false,
    path="EvoLinear.EvoLinearRegressor",
    descr=EvoLinearRegressor_desc)