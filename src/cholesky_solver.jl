
struct CholeskySolver{T,D}
    constraint_blocks::Vector{ConstraintBlock{T,D}}
    shur_blocks::Vector{BlockTriangular3{<:Any,<:Any,<:Any,T}}
    chol_blocks::Vector{BlockTriangular3{<:Any,<:Any,<:Any,T}}
end

function CholeskySolver(prob::Problem)
    conSet = get_constraints(prob)
    blocks = TrajOptCore.build_constraint_blocks(conSet)
    shur_blocks = build_shur_factors(blocks)
    chol_blocks = build_shur_factors(blocks)
    CholeskySolver(blocks, shur_blocks, chol_blocks)
end
