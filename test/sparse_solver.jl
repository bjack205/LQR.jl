using SuiteSparse
using ForwardDiff
import LQR: Primals

include("problems.jl")

prob = DoubleIntegrator(3,101, dense_cost=false)
rollout!(prob)

n,m,N = size(prob)
NN = N*n + (N-1)*m
P = sum(num_constraints(prob))
D = spzeros(P,NN)
d = zeros(P)
cons = get_constraints(prob)
cinds = LQR.gen_con_inds(get_constraints(prob))
zinds = [(k-1)*(n+m) .+ (1:n+m) for k = 1:N]
zinds[end] = (N-1)*(n+m) .+ (1:n)

# Test initial condition constraint
C,c = TrajOptCore.gen_convals(D,d,cinds[1],zinds,cons[1],cons.inds[1])
size(C) == (1,1)
size(C[1]) == (n,n)
TrajOptCore.get_dims(cons[1],n+m) == (n,m)
TrajOptCore.widths(cons[1],n,m) == (n,)
conval = ConVal(n,m, cons[1], cons.inds[1], C, c)
conval.jac[1] .= 1
D[1:n,1:n] == ones(n,n)

# Test dynamics constraints
C,c = TrajOptCore.gen_convals(D, d, cinds[end], zinds, cons[end], cons.inds[end])
C[1,1] .= 2
D[n+1:2n,1:n+m] == fill(2,n,n+m)
C[1,2] .= 3
D[n+1:2n,n+m+1:2n+2m] == [fill(3,n,n) zeros(n,m)]
typeof(cinds)

# Create constraint set
conSet = LQR.SparseConstraintSet(prob.model, prob.constraints)
evaluate!(conSet, prob.Z)
jacobian!(conSet, prob.Z)

solver = CholeskySolver(prob)
LQR.update!(solver)
D,d = LQR.get_linearized_constraints(solver)
D ≈ conSet.D
d ≈ conSet.d

# Create view costs
G = spzeros(NN,NN)
g = zeros(NN)
vobj = Objective([LQR.QuadraticViewCost(G, g, cost, i) for (i,cost) in enumerate(prob.obj)])
vobj[1]
G[1:n,1:n] == prob.obj[1].Q
G[end-n+1:end, end-n+1:end] == prob.obj[end].Q
G[n .+ (1:m), n.+ (1:m)] == prob.obj[1].R
g[1:n] == prob.obj[1].q
g[n .+ (1:m)] == prob.obj[1].r

J = Objective([LQR.QuadraticViewCost(
		G, g, QuadraticCost{Float64}(n, m, terminal=(k==N)),k)
		for k = 1:N])
cost_expansion!(J, prob.obj, prob.Z)
G0,g0 = LQR.get_cost_expansion(solver)
G0 ≈ G
g ≈ g
dt = prob.Z[1].dt
G[1:n,1:n] == prob.obj[1].Q*dt
G[end-n+1:end, end-n+1:end] == prob.obj[end].Q
G[n .+ (1:m), n.+ (1:m)] == prob.obj[1].R*dt
g[1:n] == (prob.obj[1].Q*prob.x0 + prob.obj[1].q)*dt
g[n .+ (1:m)] == prob.obj[1].r*dt


# Build solver
solver0 = CholeskySolver(prob)
LQR.update!(solver0)
solver0.conSet.convals[1].vals

solver = LQR.SparseSolver(prob)
LQR.update!(solver)

solver.conSet.D ≈ D
solver.conSet.d ≈ d

solver.G == G
solver.g == g

solver.J2[1].q == solver.J[1].q
solver.J2[1].Q == solver.J[1].Q

solver.Dblocks[1] == [solver.conSet.errvals[1].jac[1] zeros(n,m);
	solver.conSet.errvals[4].jac[1]]
solver.Dblocks[2] == [solver.conSet.errvals[4].jac[1,2] zeros(n,m);
	solver.conSet.errvals[2].jac[2];
	solver.conSet.errvals[4].jac[2]]
solver.Dblocks[3] == [solver.conSet.errvals[4].jac[2,2] zeros(n,m);
	solver.conSet.errvals[2].jac[3];
	solver.conSet.errvals[4].jac[3]]
solver.Dblocks[N] == [solver.conSet.errvals[4].jac[N-1,2];
	solver.conSet.errvals[3].jac[1]]

inds = solver.Dblocks[2].indices
(solver.conSet.D')[inds[2],inds[1]] ≈ solver.Dblocks[2]'

LQR.calc_Ginv!(solver)
solver.Ginv*solver.Z.Z ≈ solver.G\solver.Z.Z
nnz(solver.Ginv) == NN
# @btime LQR.calc_Ginv!($solver)

# LQR.calc_DH!(solver)
# @btime LQR.calc_DH!($solver)
# solver.DH.parent ≈ G\D'
# solver.δZ ≈ G\g

LQR._solve!(solver)
LQR._solve!(solver0)
dz0 = LQR.get_step(solver0)
dz = solver.δZ.Z
dz ≈ dz0
##

@btime LQR._solve!($solver)
@btime LQR._solve!($solver0)

z = prob.Z[1]
initial_trajectory!(solver, prob.Z)
merit = TrajOptCore.L1Merit(1.0)
ϕ = merit(solver)
ϕ′ = TrajOptCore.derivative(merit, solver)
ls = TrajOptCore.SimpleBacktracking()
crit = TrajOptCore.WolfeConditions()

LQR.update!(solver)
ϕ(0)
ϕ′(0)
norm(solver.g - solver.conSet.D'solver.λ)
max_violation(solver)
LQR._solve!(solver)
TrajOptCore.line_search(ls, crit, ϕ, ϕ′)
copyto!(solver.Z.Z, solver.Z̄.Z)

ϕ(0)
ϕ′(0)
LQR.update!(solver)
norm(solver.g + solver.conSet.D'solver.λ)
solver.g

Z = copy(solver.Z)
dZ = copy(solver.δZ)

l1 = TrajOptCore.L1Merit(1.0)
LQR.l1merit(solver)
l1(solver, 0)

LQR.l1grad(solver)
TrajOptCore.derivative(l1, solver, 0)

LQR.line_search(solver, l1)
solver.Z̄.Z == solver.Z.Z .+ solver.δZ.Z

solver.g + solver.conSet.D'solver.λ

@btime LQR.l1merit($solver)
@btime $l1($solver, 0)
@btime LQR.l1grad($solver)
@btime TrajOptCore.derivative($l1, $solver, 0)

TrajOptCore.norm_grad(solver.J, 1)
@btime TrajOptCore.norm_grad($solver.J, 2)
@btime norm($solver.g, 2)


# Check gradient with ForwardDiff
_zinds = [SVector{length(ind)}(ind) for ind in zinds]
function mycost(Z)
	_Z = [StaticKnotPoint(prob.Z[k], Z[_zinds[k]]) for k = 1:N]
	J = zeros(eltype(Z), N)
	TrajOptCore.cost!(prob.obj, _Z, J)
	return sum(J)
end
x = copy(solver.Z)
p = copy(solver.δZ)
mycost(x.Z) ≈ cost(solver.obj, solver.Z.Z_)
cost_gradient!(solver.J, solver.obj, solver.Z.Z_)
ForwardDiff.gradient(mycost, x.Z) ≈ solver.g
TrajOptCore.dgrad(solver.J, p.Z_) ≈ solver.g'p.Z

# Check directional derivative of cost with finite diff
t = 1e-8
f0 = LQR.l1merit(solver, x, 0)
f0 ≈ mycost(x.Z)
(x + t*p).Z ≈ (x.Z + t*p.Z)
f1 = LQR.l1merit(solver, x + t*p, 0)
(f1-f0)/t
abs(LQR.l1grad(solver, solver.Z, solver.δZ, 0) - (f1-f0)/t) < 1e-4

# Check
Z = copy(solver.Z)
dZ = copy(solver.δZ)
mul!(solver.r, D, p.Z)
TrajOptCore.norm_dgrad(d, solver.r) ≈ TrajOptCore.norm_dgrad(solver.conSet, dZ.Z_)
@btime LQR.l1grad($solver, $Z, $dZ)
@btime TrajOptCore.norm_dgrad($solver.conSet, $dZ.Z_)

l1(solver)
TrajOptCore.derivative(l1, solver)
LQR.line_search(solver, l1)
f0 = LQR.l1merit(solver, x)
f1 = LQR.l1merit(solver, x + t*p)
(f1-f0)/t
TrajOptCore.norm_violation(solver.conSet)
LQR.l1grad(solver)
solver.conSet.d

Z = LQR.Primals(prob)
copyto!(Z, prob.Z)
dZ = LQR.Primals(prob)
copyto!(dZ, solver.δZ)

struct MySolverWrapper{S}
	solver::S
end

mutable struct MeritFun{n,m,T}
	merit::Function
	dgrad::Function
	x0::Primals{n,m,T}
	p::Primals{n,m,T}
	x::Primals{n,m,T}
	α::T
end

function update!(ϕ::MeritFun, α)
	if α ≈ 0
		return ϕ.x0
	elseif !(α ≈ ϕ.α)
		ϕ.x.Z .= ϕ.x0.Z .+ α*ϕ.p.Z  # calculate new candidate
		ϕ.α = α
	end
	return ϕ.x
end

function (ϕ::MeritFun)(α::Real)
	x = update!(ϕ, α)
	ϕ.merit(x)
end

function deriv(ϕ::MeritFun, α::Real)
	x = update!(ϕ, α)
	ϕ.dgrad(x)
end


Z = copy(solver.Z)
dZ = copy(solver.δZ)

meritfun(x) = LQR.l1merit(solver, x)
deriv(x) = LQR.l1grad(solver, x)
merit = MeritFun(meritfun, deriv, Z, dZ, Primals(prob), 1.0)
merit(0)
merit(1)
deriv(merit, 1)
LQR.dgrad(solver.J, dZ.Z_)
solver.J.J
cost_gradient!(solver.J, solver.obj, solver.Z.Z_)
dZ.Z

struct L1Merit{C}
    obj::Objective
    conSet::C
	μ::Float64
end

function TrajOptCore.evaluate!(merit::L1Merit, Z::Traj)
    TrajOptCore.cost!(merit.obj, Z)
    J = TrajOptCore.get_J(merit.obj)::Vector{Float64}

    evaluate!(merit.conSet, Z)
	c = TrajOptCore.norm_violation(merit.conSet, 1)::Float64
    sum(J) + c
end

function dgrad(merit::L1Merit, Z::Traj, dz)
	D,d = merit.conSet.D, solver.conSet.d
end

merit = MyMerit(prob.obj, solver.conSet, 1.0)
evaluate!(merit, prob.Z)
Z = prob.Z
@btime evaluate!($merit, $Z)
TrajOptCore.norm_violation(merit.conSet, 1)

struct BacktrackSimple
end

function line_search(ls::BacktrackSimple, merit, x, p)

end

struct Wolfe{T}
	c1::T
	c2::T
end

function sufficient_decrease(condition::Wolfe, merit, x0, p, α)
	x = x0 .+ α*p
	merit(x) ≤ merit(x0) + condition.c1*α*dgrad(merit, x0, p)
end

function curvature(condition::Wolfe, merit, x0, p, α)
	x = x0 .+ α*p
	dgrad(merit, x, p) ≥ condition.c2*dgrad(merit, x0, p)
end
