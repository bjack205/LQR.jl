include("problems.jl")

prob =  DoubleIntegrator(3,11)

J = Objective(TrajOptCore.QuadraticCostFunction.(prob.obj.cost))
J = TrajOptCore.QuadraticObjective(prob.obj)
Jinv = Objective(inv.(J))

n,m,N = size(prob)
prob.obj[N-1]
conSet = get_constraints(prob)
blocks = TrajOptCore.ConstraintBlocks(conSet)
blocks[2].D1
blocks[end-1]
solver = LQR.CholeskySolver(prob)

# evaluate!(conSet, prob.Z)
# jacobian!(conSet, prob.Z)
# con = LQR.link_jacobians(blocks, conSet[3], zeros(Int,N))
# conSet[2].∇c[1]
# con.∇c[1] .= 1;
# blocks[2].Y[n+1,:1:n] == ones(n)
#
# con = LQR.link_jacobians(blocks, conSet[1], zeros(Int,N))
# con.∇c[1] .= 3
# blocks[1].Y[1:n,1:n] == fill(3,n,n)
# blocks[1].Y[1:n,n .+ (1:m)] == zeros(n,m)
#
# con = LQR.link_jacobians(blocks, conSet[2], zeros(Int,N))
# con.∇c[1] .= 2
# con.∇c[1,2] .= -2
# blocks[1].Y[end-n+1:end, :] == fill(2,n,n+m)
# blocks[2].Y[1:n,:] == fill(-2,n,n+m)
A = rand(10,10)
A = A'A
inv(A)
LAPACK.potri!('L',A)
LAPACK.sytri!('U', A, zeros(Int,10))
A

block = blocks[2]
J = prob.obj[2]
res = LQR.BlockTriangular3(n,dims(block)[2],n)
LQR.shur!(res, J, block)
@btime LQR.shur!($res, $J, $block)

# Set up the problem
prob =  DoubleIntegrator(2,11)
n,m,N = size(prob)
P = sum(num_constraints(prob))
NN = num_vars(prob)
rollout!(prob)
conSet = get_constraints(prob)

# Create Jacobian blocks
blocks = TrajOptCore.ConstraintBlocks(conSet)
evaluate!(blocks, prob.Z)
jacobian!(blocks, prob.Z)

# Create Shur factor blocks
iobj = Objective(inv.(prob.obj.cost))
F = LQR.build_shur_factors(blocks)
Ft = LQR.build_shur_factors(blocks, :U)
LQR.calculate_shur_factors!(F, iobj, blocks)
LQR.calculate_shur_factors!(Ft, iobj, blocks)
F[10].D ≈ (Ft[10].D')
F[10].F ≈ (Ft[10].F')
F[10].B ≈ (Ft[10].B')

# Create full array versions
S = zeros(P,P)
d_ = zeros(P)
λ = zeros(P)
LQR.copy_shur_factors!(S, d_, λ, F)
S_ = Symmetric(S, :L)
d_

D = zeros(P, NN)
d = zeros(P)
vblocks = TrajOptCore.ConstraintBlocks(D,d,blocks)
evaluate!(vblocks, prob.Z)
jacobian!(vblocks, prob.Z)
d ≈ d_

H = zeros(NN,NN)
LQR.build_H!(H, prob.obj)

S0 = Symmetric(D*(H\D'))
S0 ≈ S_

# Calculate cholesky factorization
LQR.calculate_shur_factors!(F, iobj, blocks)
LQR.calculate_shur_factors!(Ft, iobj, blocks)
L = LQR.build_shur_factors(blocks)
U = LQR.build_shur_factors(blocks, :U)

cholesky!(L, F)
LQR.Ucholesky!(U, Ft)
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
copyto!(sol, prob.Z)
Z0 = copy(sol.Z)
LQR.calculate_primals!(sol, prob.obj, L, blocks)
sol.Z
A = conSet[2].con.A
# sol.Z .+= Z0
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
LQR.solve!(sol, solver)
@btime LQR.solve!($sol, $solver)

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
