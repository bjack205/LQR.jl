using SuiteSparse
include("problems.jl")

prob = DoubleIntegrator(3,101, dense_cost=true)
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
solver.Ginv*solver.Z ≈ solver.G\solver.Z
nnz(solver.Ginv) == NN
@btime LQR.calc_Ginv!($solver)

LQR.calc_DH!(solver)
@btime LQR.calc_DH!($solver)
solver.DH.parent ≈ G\D'
solver.δZ ≈ G\g

dz = LQR._solve!(solver)
LQR._solve!(solver0)
dz0 = LQR.get_step(solver0)
dz ≈ dz0

@btime LQR._solve!($solver)
@btime LQR._solve!($solver0)
