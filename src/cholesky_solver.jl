
struct CholeskySolver{n,m,nm,T,D}
    obj::Objective
    J::Objective
    Jinv::Objective{<:QuadraticCost}
    constraint_blocks::Vector{ConstraintBlock{T,Vector{T},Vector{T},Matrix{T},Matrix{T},D}}
    shur_blocks::Vector{<:BlockTriangular3{<:Any,<:Any,<:Any,T}}
    chol_blocks::Vector{<:BlockTriangular3{<:Any,<:Any,<:Any,T}}
    δZ::Vector{KnotPoint{T,n,m,nm}}
    Z::Vector{KnotPoint{T,n,m,nm}}
    Z̄::Vector{KnotPoint{T,n,m,nm}}
end

function CholeskySolver(prob::Problem)
    J = TrajOptCore.QuadraticObjective(prob.obj)
    Jinv = Objective(inv.(J))

    conSet = get_constraints(prob)
    blocks = TrajOptCore.ConstraintBlocks(conSet)
    shur_blocks = build_shur_factors(blocks, :U)
    chol_blocks = build_shur_factors(blocks, :U)
    evaluate!(blocks, prob.Z)
    jacobian!(blocks, prob.Z)
    iobj = Objective(inv.(prob.obj.cost))
    CholeskySolver(iobj, blocks, shur_blocks, chol_blocks)
end

function solve!(sol, solver::CholeskySolver)
    for i = 1:10
        # Update constraints
        evaluate!(blocks, solver.Z)
        jacobian!(blocks, solver.Z)

        # Update cost function
        cost_expansion!(solver.J, solver.obj, solver.Z)
    end
end

function _solve!(solver::CholeskySolver)
    calculate_shur_factors!(solver.shur_blocks, solver.obj, solver.constraint_blocks)
    # chol = solver.chol_blocks
    # cholesky!(chol, solver.shur_blocks)
    chol = solver.shur_blocks
    cholesky!(solver.shur_blocks)
    forward_substitution!(chol)
    backward_substitution!(chol)
    calculate_primals!(sol, solver.obj, chol, solver.constraint_blocks)
end

function calculate_primals!(sol::LQRSolution, iobj::Objective, chol, blocks::ConstraintBlocks)
    N = length(blocks)
    for k = 1:N
        Jinv = iobj.cost[k]
        Hinv,g = hessian(Jinv), gradient(Jinv)
        z = sol.Z_[k].z
        if k == 1
            calc_primals!(z, chol[k], blocks[k])
            # sol.Z_[k].z .+= -Hinv*(blocks[k].C'chol[k].μ .+ blocks[k].D1'chol[k].λ)
        else
            calc_primals!(z, chol[k], chol[k-1], blocks[k])
            # sol.Z_[k].z .+= -Hinv*(blocks[k].D2'chol[k-1].λ .+ blocks[k].C'chol[k].μ
            #     .+ blocks[k].D1'chol[k].λ)
        end
        mul!(z, -Hinv, z)
    end
end

function calc_primals!(z, chol, block)
    mul!(z, block.D1', chol.λ)
    mul!(z, block.C', chol.μ, 1.0, 1.0)
    # mul!(z, -Hinv, z)
    # z .= -Hinv*(block.C'chol.μ .+ block.D1'chol.λ)
end

function calc_primals!(z, chol, chol_prev, block)
    calc_primals!(z, chol, block)
    mul!(z, block.D2', chol_prev.λ, 1.0, 1.0)
    # mul!(z, -Hinv, z)
    # z .= -Hinv*(block.C'chol.μ .+ block.D1'chol.λ)
end
