
struct CholeskySolver{T}
    constraint_blocks::Vector{ConstraintBlock{T}}
    shur_blocks::Vector{BlockTriangular3{<:Any,<:Any,<:Any,T}}
    chol_blocks::Vector{BlockTriangular3{<:Any,<:Any,<:Any,T}}
    constraints::ConstraintSet{T}
end

function CholeskySolver(prob::Problem)
    conSet = get_constraints(prob)
    blocks = build_jacobian_blocks(conSet)
    link_jacobians(conSet)
end
