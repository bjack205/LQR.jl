using Test
include("problems.jl")

import LQR: ConstraintBlock, ConstraintBlocks

# Set up the problem
prob = DoubleIntegrator(3,101)
model = prob.model
n,m,N = size(prob)
P = sum(num_constraints(prob))
NN = num_vars(prob)
rollout!(prob)
conSet = get_constraints(prob)
sort!(conSet)

# Calculate the inverted cost
Jinv = LQR.InvertedQuadratic.(prob.obj.cost)
LQR.InvertedQuadratic(prob.obj.cost[end])
LQR.update_cholesky!(Jinv, prob.obj)
# @btime LQR.update_cholesky!($Jinv, $prob.obj)


# Create Jacobian blocks
blocks = ConstraintBlocks(model, conSet)
@test size(blocks[1].Y) == (2n,n+m)
@test size(blocks[1].D2,1) == 0
@test size(blocks[1].D1,1) == n
@test size(blocks[N].Y) == (2n,n)
@test size(blocks[N].D2,1) == n
@test size(blocks[N].D1,1) == 0
@test size(blocks[N].C,1) == conSet.p[N]
@test size(blocks[N÷2].C,1) == conSet.p[N÷2]-n
cinds = LQR.gen_con_inds(conSet, :by_block)
@test cinds[1][1] == 1:n
@test cinds[end][1] == n .+ (1:n)
@test cinds[end][2] == 2:n+1
@test cinds[2][2] == 1:1
@test cinds[3][1] == 1:n

cvals = map(enumerate(zip(conSet))) do (i,(inds,con))
    C,c = TrajOptCore.gen_convals(blocks, cinds[i], con, inds)
    ConVal(n,m, con, inds, C, c)
end
cvals[1].jac[1] .= 1
@test blocks[1].C[:,1:n] == ones(n,n)
cvals[4].jac[1,1] .= 2
@test blocks[1].D1 == fill(2,n,n+m)
cvals[4].jac[2,1] .= 3
@test blocks[2].D1 == fill(3,n,n+m)
cvals[4].jac[1,2] .= 4
@test blocks[2].D2 == fill(4,n,n+m)
cvals[4].jac[end,1] .= -2
@test blocks[end-1].D1 == fill(-2,n,n+m)
cvals[4].jac[end,2] .= -1
@test blocks[end].D2 == fill(-1,n,n)
cvals[1].vals[1] .= 1.5
@test blocks[1].c  == fill(1.5,n)
cvals[4].vals[1] .= 2.5
@test blocks[1].d == fill(2.5,n)

conSet = LQR.BlockConstraintSet(model, get_constraints(prob))
evaluate!(conSet, prob.Z)
jacobian!(conSet, prob.Z)

# Create Shur factor blocks
F = LQR.build_shur_factors(conSet)
Ft = LQR.build_shur_factors(conSet, :U)
LQR.calculate_shur_factors!(F, Jinv, conSet.blocks)
LQR.calculate_shur_factors!(Ft, Jinv, conSet.blocks)
@test F[10].D ≈ (Ft[10].D')
@test F[10].F ≈ (Ft[10].F')
@test F[10].B ≈ (Ft[10].B')


# Create full array versions
S = zeros(P,P)
d_ = zeros(P)
λ = zeros(P)
LQR.copy_shur_factors!(S, d_, λ, F)
S_ = Symmetric(S, :L)

D = zeros(P, NN)
d = zeros(P)
@which LQR.copy_blocks!(D, d, conSet.blocks)
@test d ≈ d_

H = zeros(NN,NN)
LQR.build_H!(H, prob.obj)

S0 = Symmetric(D*(H\D'))
@test S0 ≈ S_

# Calculate cholesky factorization
LQR.calculate_shur_factors!(F, Jinv, conSet.blocks)
LQR.calculate_shur_factors!(Ft, Jinv, conSet.blocks)
L = LQR.build_shur_factors(conSet)
U = LQR.build_shur_factors(conSet, :U)

Ft[1].B
solver.shur_blocks[1].B
Jinv[1].chol.M
solver.Jinv[1].chol.M
cholesky!(L, F)
cholesky!(U, Ft)
U[1].B
F[1].B
U[1].D'U[1].D
LQR.forward_substitution!(L)
LQR.forward_substitution!(U)
# LQR.Ucholesky!(Ft)

# @btime cholesky!($L, $F)
# @btime LQR.Ucholesky!($U, $Ft)
# @btime LQR.Ucholesky!($Ft)

L_ = zeros(P,P)
U_ = zeros(P,P)
U2_ = zeros(P,P)
# LQR.copy_shur_factors!(U_,d_,λ, Ft)
LQR.copy_shur_factors!(L_,d_,λ, L)
LQR.copy_shur_factors!(U_,d_,λ, U)
L_ ≈ U_'
L_ ≈ cholesky(S0).L
U_ ≈ cholesky(S0).U
cholesky(S0).L\d ≈ λ

λ_ = copy(λ)
LQR.backward_substitution!(L)
LQR.backward_substitution!(U)
# @btime LQR.backward_substitution!($U)
LQR.copy_shur_factors!(L_,d_,λ, L)
LQR.copy_shur_factors!(U_,d_,λ, U)
cholesky(S0)\d ≈ λ

sol = LQRSolution(prob)
rollout!(prob)
copyto!(sol, prob.Z)
Z0 = copy(sol.Z)
Z̄ = [StaticKnotPoint(z) for z in prob.Z]
δZ = [LQR.MutableKnotPoint(n,m, (@MVector zeros(n+m)), z.dt, z.t) for z in prob.Z]
LQR.calculate_primals!(δZ, Jinv, L, conSet.blocks)
Z̄ .= prob.Z .+ δZ

max_violation(conSet)
evaluate!(conSet, Z̄)
max_violation(conSet)
δZ[1]
solver.δZ[1]


cons = get_constraints(prob)
A = cons[2].A
sol.Z .+= Z0
r_con = map(2:N-1) do k
    (A*sol.X[k])[1]
end
norm(r_con,Inf)


# Use CholeskySolver
prob =  DoubleIntegrator(2,11)
n,m,N = size(prob)
rollout!(prob)
sol = LQRSolution(prob)
copyto!(sol, prob.Z)
Z0 = copy(sol.Z)
solver = LQR.CholeskySolver(prob)
LQR._solve!(sol, solver)
@btime LQR._solve!($sol, $solver)

sol.Z .+= Z0
A = prob.constraints[2].con.A
r_con = map(2:N-1) do k
    (A*sol.X[k])[1]
end
norm(r_con,Inf)

# Time solve
prob =  DoubleIntegrator(2,11)
rollout!(prob)
sol = LQRSolution(prob)
solver = LQR.CholeskySolver(prob)
@btime LQR.calculate_shur_factors!($solver.shur_blocks, $solver.obj, $solver.constraint_blocks)
@btime cholesky!($solver.chol_blocks, $solver.shur_blocks)
@btime cholesky!($solver.shur_blocks)
@btime LQR.forward_substitution!($solver.chol_blocks)
@btime LQR.backward_substitution!($solver.chol_blocks)
@btime LQR.calculate_primals!($sol, $solver.obj, $solver.chol_blocks, $solver.constraint_blocks)

# Test cholesky
solver = LQR.CholeskySolver(prob)
LQR.calculate_shur_factors!(solver.shur_blocks, solver.obj, solver.constraint_blocks)
F = solver.shur_blocks
L = solver.chol_blocks
cholesky!(L,F)

A = cholesky(F[1].A)
cholesky!(L[1], F[1], A)
a = tr(L[1].C)
LQR.cholesky2!(L[1], F[1], A)
tr(L[1].C) ≈ a
@btime cholesky!($(L[1]),$(F[1]),UpperTriangular($A))

A = rand(10,10)
B = Matrix(LowerTriangular(rand(10,10)))
A/B ≈ (B'\A')'
B'
LAPACK.trtrs!('L','T','N',B,A)
@which A/B
typeof(transpose(A))

struct AAt{T}
    A::Matrix{T}
    At::Transpose{T,Matrix{T}}
    AAt(A::Matrix{T}) where T = new{T}(A,transpose(A))
end
At = AAt(A)
At.At[2] = 1
@btime $At.At[3] = 3

A[3] = 4
A


A = LQR.chol!(F[1].A)
LQR.chol!(L[1], F[1], A)
@btime LQR.chol!($(L[1]), $(F[1]), $A)



n,m = size(prob)
p = 1
A,B = linearize(integration(prob), prob.model, prob.Z[1])
Ā = SizedMatrix{n,n}(1.0I(n))
B̄ = SizedMatrix{n,m}(zeros(n,m))
C = rand(SizedMatrix{p,n})
D = rand(SizedMatrix{p,m})
d = SizedVector{p}(zeros(p))
Q = Diagonal(@SVector rand(n))
R = Diagonal(@SVector rand(m))

J = QuadraticCost(Q, R)
con = LQR.JacobianBlock(Ā,B̄,C,D,d,A,B)
res = LQR.BlockTriangular3{Float64}(n,m,p)
@time LQR.shur!(res,J,con)
@btime LQR.shur!($res, $J, $con)

sz = size(con)
E = SizedMatrix{sz...}(zeros(sz))
E = SizedMatrix{sz...}([con.Ā con.B̄; con.C con.D; con.A con.B])
H = SizedMatrix{n+m,n+m}(cat(Q,R,dims=[1,2]))
EEt = SizedMatrix{15,15}(zeros(15,15))
EEt .= E*H*E'
ix = 1:n
ip = 1:p
EEt[ix,ix] ≈ res.A
EEt[n .+ ip, n .+ ip] ≈ res.B
EEt[n .+ ip, ix] ≈ res.D
EEt[n .+ ip, ix] ≈ res.D
@btime $EEt .= $E*$H*$E'



prob = DoubleIntegrator(3, constrained=true)
p = num_constraints(prob)
n,m,N = size(prob)
NN =  num_vars(prob)
P = sum(p)
D = zeros(P,NN)
LQR.copy_jacobian!(D,prob.con[1])
num_constraints(conSet)
conSet

build_jacobian_blocks(conSet)



TrajOptCore.contype(conSet.constraints[1])
@btime TrajOptCore.findfirst_constraint($LinearConstraint, $conSet)
LinearConstraint <: AbstractConstraint{S,W,P} where {S,W,P}
conSet.constraints[1].con isa LinearConstraint

# function build_jacobian_blocks(conSet::ConstraintSet)
#     n,m = size(conSet)
#     N = length(conSet.p)
#     p = num_constraints(conSet)
#     if conSet[1].con isa DynamicsConstrain
# end
# @btime copy_jacobian!($(blocks[2]), $conSet, 2)
#
# function copy_jacobian!(block::JacobianBlock{n,m,p},
#         con::ConstraintVals{<:Any,State}, k::Int, off::Int) where {n,m,p}
#     if k ∈ con.inds
#         ip = SVector{p}(1:p)
#         uview(block.C.data,off .+ ip,:) .= con.∇c[k]
#     end
# end
# @btime copy_jacobian!($(blocks[2]), $(conSet[2]), 2, 0)
#
# function copy_jacobian!(blocks, con)
#     for (i,k) in enumerate(con.inds)
#         blocks[k].C .= con.∇c[i]
#     end
# end
# copy_jacobian!(blocks, conSet[2])
# @btime copy_jacobian!($blocks, $(conSet[2]))
#
# function copy_dynamics_jacobian!(block::JacobianBlock{n,m,p},
#         dyn_con::ConstraintVals{<:Any,<:Any,<:DynamicsConstraint{<:RobotDynamics.Implicit}},
#         k::Int) where {n,m,p}
#     ix = SVector{n}(1:n)
#     iu = SVector{m}(1:m)
#     if k ∈ dyn_con.inds
#         ∇f = dyn_con.∇c[k]
#         block.Ā .= Diagonal(@SVector ones(n))
#         block.B̄ .= 0
#         block.A .= ∇f[ix,ix]
#         block.B .= ∇f[ix,iu]
#     end
# end
