include("problems.jl")

prob =  DoubleIntegrator(3,11)
TrajOptCore.num_constraints!(prob.constraints)
num_constraints(prob)
LQR.get_linearized_constraints(prob.constraints)
LCRProblem(prob)
prob.constraints.p
n,m,N = size(prob)
P = sum(num_constraints(prob))
NN = num_vars(prob)
rollout!(prob)
conSet = get_constraints(prob)
blocks = LQR.build_jacobian_blocks(conSet)
size(blocks[end].Y) == (n,n)

con = LQR.link_jacobians(blocks, conSet[3], zeros(Int,N))
conSet[2].∇c[1]
con.∇c[1] .= 1;
blocks[2].Y[n+1,:1:n] == ones(n)

con = LQR.link_jacobians(blocks, conSet[1], zeros(Int,N))
con.∇c[1] .= 3
blocks[1].Y[1:n,1:n] == fill(3,n,n)
blocks[1].Y[1:n,n .+ (1:m)] == zeros(n,m)

con = LQR.link_jacobians(blocks, conSet[2], zeros(Int,N))
con.∇c[1] .= 2
con.∇c[1,2] .= -2
blocks[1].Y[end-n+1:end, :] == fill(2,n,n+m)
blocks[2].Y[1:n,:] == fill(-2,n,n+m)

block = blocks[2]
J = prob.obj[2]
res = LQR.BlockTriangular3(n,dims(block)[2],n)
LQR.shur!(res, J, block)
@btime LQR.shur!($res, $J, $block)

# Set up the problem
prob =  DoubleIntegrator(3,31)
n,m,N = size(prob)
P = sum(num_constraints(prob))
NN = num_vars(prob)
rollout!(prob)
conSet = get_constraints(prob)

# Create Jacobian blocks
blocks = LQR.build_jacobian_blocks(conSet)
blocks[1]
LQR.link_jacobians(blocks, conSet)
evaluate!(conSet, prob.Z)
jacobian!(conSet, prob.Z)

# Create Shur factor blocks
F = LQR.build_shur_factors(conSet)
LQR.calculate_shur_factors!(F, prob.obj, blocks)

# Create full array versions
S = zeros(P,P)
h = zeros(P)
λ = zeros(P)
LQR.copy_shur_factors!(S, h, λ, F)
LQR.copy_block!(S,h,λ,F[1],0)
S_ = Symmetric(S, :L)

D = zeros(P, NN)
D_ = LQR.jacobian_views!(D,conSet)
LQR.copy_jacobians!(D_, blocks)

H = zeros(NN,NN)
LQR.build_H!(H, prob.obj)

S0 = D*H*D'
S0 ≈ S_

p = conSet.p
d = [SizedVector{p[k]}(zeros(p[k]))  for k = 1:N]
LQR.copy_vals!(d, conSet)

# Calculate cholesky factorization
L = LQR.build_shur_factors(conSet)
L[1].c .= prob.x0
cholesky!(L, F)
LQR.forward_substitution!(L)

L_ = zero(S0)
LQR.copy_shur_factors!(L_,h,λ, L)
L_ ≈ cholesky(S0).L
cholesky(S0).L\h ≈ λ

LQR.backward_substitution!(L)
LQR.copy_shur_factors!(L_,h,λ, L)
cholesky(S0)\h ≈ λ


D = Diagonal(@SVector ones(40))
Hv = view(H,1:40,1:40)
Hv = view(H,diag(LinearIndices(H)[1:40,1:40]))
@btime diag($Hv) .= diag($D)
@btime copyto!($Hv, $D)

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
