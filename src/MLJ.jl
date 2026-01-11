function MMI.fit(learner::EvoLinearRegressor, verbosity::Int, A, y, w=nothing)
    A = isa(A, AbstractMatrix) ? Tables.columntable(Tables.table(A)) : Tables.columntable(A)
    m, cache = init(learner; features=A, target=y, weight=w)
    while m.info[:nrounds] < learner.nrounds
        fit!(m, cache, learner)
    end
    report = (coef=m.coef, bias=m.bias, names=m.info[:feature_names])
    return m, cache, report
end

function okay_to_continue(config, m, cache)
    return config.nrounds - m.info[:nrounds] >= 0
end

function MMI.update(learner::EvoLinearRegressor, verbosity::Integer, m, cache, A, y, w=nothing)
    if okay_to_continue(learner, m, cache)
        while m.info[:nrounds] < learner.nrounds
            fit!(m, cache, learner)
        end
        report = (coef=m.coef, bias=m.bias, names=m.info[:feature_names])
    else
        m, cache, report = fit(learner, verbosity, A, y, w)
    end
    return m, cache, report
end

predict(::EvoLinearRegressor, m::EvoLinearModel, A) = m(A)

# For EarlyStopping.jl supportm
MMI.iteration_parameter(::Type{<:EvoLinearTypes}) = :nrounds

# Metadata
MMI.metadata_pkg.(
    (EvoLinearRegressor),
    name="EvoLinear",
    uuid="ab853011-1780-437f-b4b5-5de6f4777246",
    url="https://github.com/jeremiedb/EvoLinear.jl",
    julia=true,
    license="MIT",
    is_wrapper=false,
)

MMI.metadata_model(
    EvoLinearRegressor,
    input_scitype=Union{
        MMI.Table(MMI.Continuous, MMI.Count, MMI.OrderedFactor),
    },
    target_scitype=AbstractVector{<:MMI.Continuous},
    supports_weights=true,
    path="EvoLinear.EvoLinearRegressor",
)
