include("problems.jl")

prob = DoubleIntegrator(3,101)
rollout!(prob)
solver = CholeskySolver(prob)
num_constraints(prob)
LQR.update!(solver)
P = sum(solver.conSet.p)
NN = num_vars(solver)

# Compute shur compliment
LQR.calculate_shur_factors!(solver.shur_blocks, solver.Jinv, solver.conSet.blocks)

# Compare againt block calculation
D,d = LQR.get_linearized_constraints(solver)
H,g = LQR.get_cost_expansion(solver)
S,r = LQR.get_shur_factors(solver)
S ≈ D*(H\D')
D*(H\g) - d ≈ r

# Compute the cholesky factorization
cholesky!(solver.chol_blocks, solver.shur_blocks)
U = LQR.get_cholesky(solver)
cholesky(S).U ≈ U
U'U ≈ S

# Compute the forward and backward_substitution to compute multipliers
LQR.forward_substitution!(solver.chol_blocks)
LQR.backward_substitution!(solver.chol_blocks)
λ = LQR.get_multipliers(solver)
λ ≈ -S\r

# Compute the primal step
LQR.calculate_primals!(solver.δZ, solver.Jinv, solver.chol_blocks, solver.conSet.blocks)
dZ = LQR.get_step(solver)
dZ ≈ -H\(D'λ + g)

# Check residuals
norm(D*dZ + d) < 1e-12
norm(H*dZ + g+ D'λ,2) <  1e-12

sol = [H D'; D zeros(P,P)]\[-g;-d]
sol[1:NN] ≈ dZ
sol[NN+1:end] ≈ λ

# Compare timing results
@btime LQR._solve!($solver);

let A = [H D'; D spzeros(P,P)], b = [-g;-d]
    @btime $A\$b
end;  # 85x slower dense, 11x slower sparse
let H = H, g = g, D = D, d = d
    @btime begin
        HD = H\D'
        S = D*HD
        λ = HD'g - d
        F = cholesky!(Symmetric(S)) # this is about 1.5x faster
        λ = ldiv!(F,λ)
        dz = H\(-D'λ + g)
    end
end;  # 42x slower
let H = H, g = g, D = D, d = d
    @btime begin
        HD = H\D'
        S = D*HD
        r = HD'g - d
        λ = Symmetric(S)\r
        dz = H\(-D'λ + g)
    end
end;  # 1.6x slower


Hg = H\g
zinds = [(1:n+m) .+ (k-1)*(n+m) for k = 1:N]
-blocks[1].c + blocks[1].C*Hg[zinds[1]] ≈ r[1:6]
-blocks[1].d + blocks[1].D1*Hg[zinds[1]] + blocks[2].D2*Hg[zinds[2]]
blocks[1].r_[3] ≈ blocks[1].D1*Hg[zinds[1]]
blocks[2].r_[1] ≈ blocks[2].D2*Hg[zinds[2]]
blocks[1].r_[3] + blocks[2].r_[1]
solver.shur_blocks[1].d
r[7:12]


max_violation(solver.conSet)
TrajOptCore.norm_violation(solver.conSet)
LQR.update_cholesky!(solver.Jinv, solver.obj)
solver.Jinv[1].chol.M
solver.Jinv[1].q
LQR.update_cholesky!(solver.Jinv, solver.J)
solver.Jinv[1].chol.M
solver.Jinv[1].q
obj2 = Objective([cost for cost in solver.J])

solver = CholeskySolver(prob)
LQR.update!(solver)

LQR._solve!(solver)
solver.J[end]
solver.obj[end]

solver.Z̄ .= solver.Z .+ solver.δZ
evaluate!(solver.conSet, solver.Z̄)
TrajOptCore.norm_violation(solver.conSet)

cost(solver.obj, solver.Z)
cost(solver.obj, solver.Z̄)



@btime $solver.Z̄ .= $solver.Z .+ $solver.δZ

@btime LQR.update!($solver)
@btime LQR._solve!($solver)

struct MyMerit
    obj::Objective
    conSet::TrajOptCore.AbstractConstraintSet
end

function TrajOptCore.evaluate!(merit::MyMerit, Z::Traj)
    TrajOptCore.cost!(merit.obj, Z)
    J = TrajOptCore.get_J(merit.obj)::Vector{Float64}

    evaluate!(merit.conSet, Z)
    sum(J)
end


merit = MyMerit(prob.obj, solver.conSet)
evaluate!(merit, prob.Z)
@btime evaluate!($merit, $(prob.Z))
@btime TrajOptCore.cost!($prob.obj, $prob.Z)
@btime TrajOptCore.stage_cost($(prob.obj.cost[1]), $(prob.Z[1]))
costfun = QuadraticCost(Diagonal(@SVector ones(6)), Diagonal(@SVector ones(3)))
@btime TrajOptCore.stage_cost($costfun, $(prob.Z[1]))

dz = solver.δZ[2]
z = solver.Z[2]
Z0 = [StaticKnotPoint(z, z.z) for z in solver.Z]
Z = solver.Z
dZ = solver.Z̄
@btime $Z0 .= $Z .+ $dZ
