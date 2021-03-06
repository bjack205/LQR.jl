using TrajOptCore
using BenchmarkTools

using SuiteSparse
struct SparseConstraintSet{T} <: TrajOptCore.AbstractConstraintSet
	convals::Vector{ConVal}
	errvals::Vector{ConVal}
	cinds::Vector{Vector{UnitRange{Int}}}
	zinds::Vector{UnitRange{Int}}
    D::SparseMatrixCSC{T,Int}               # Constraint Jacobian, by knot point
    d::Vector{T}                            # Constraint violation
	c_max::Vector{T}
end

function SparseConstraintSet(model::AbstractModel, cons::ConstraintList,
		jac_structure=:by_knotpoint)
	if !TrajOptCore.has_dynamics_constraint(cons)
		throw(ArgumentError("must contain a dynamics constraint"))
	end

	n,m = size(model)
	n̄ = RobotDynamics.state_diff_size(model)
	ncon = length(cons)
	N = length(cons.p)

	# Block sizes
	NN = N*n̄ + (N-1)*m
	P = sum(num_constraints(cons))

	# Initialize arrays
	D = spzeros(P,NN)
	d = zeros(P)

	# Create ConVals as views into D and d
	cinds = gen_con_inds(cons, jac_structure)
	zinds = [(k-1)*(n+m) .+ (1:n+m) for k = 1:N]
	useG = model isa LieGroupModel
	errvals = map(enumerate(zip(cons))) do (i,(inds,con))
		C,c = TrajOptCore.gen_convals(D, d, cinds[i], zinds, con, inds)
		ConVal(n̄, m, con, inds, C, c)
	end
	convals = map(errvals) do errval
		ConVal(n, m, errval)
	end
	errvals = convert(Vector{ConVal}, errvals)
	convals = convert(Vector{ConVal}, convals)

	SparseConstraintSet(convals, errvals, cinds, zinds, D, d, zeros(ncon))
end

@inline TrajOptCore.get_convals(conSet::SparseConstraintSet) = conSet.convals
@inline TrajOptCore.get_errvals(conSet::SparseConstraintSet) = conSet.errvals

function norm_violation(conSet::SparseConstraintSet, p=2)
	norm(conSet.d, p)
end

struct QuadraticViewCost{n,m,T} <: TrajOptCore.QuadraticCostFunction{n,m,T}
	Q::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
	R::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
	H::SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}
	q::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
	r::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
	c::T
	zeroH::Bool
	terminal::Bool
	function QuadraticViewCost(Q::SubArray, R::SubArray, H::SubArray,
		q::SubArray, r::SubArray, c::Real; checks::Bool=true, terminal::Bool=false)
		if checks
			TrajOptCore.run_posdef_checks(Q,R)
		end
		n,m = length(q), length(r)
        T = promote_type(eltype(Q), eltype(R), eltype(H), eltype(q), eltype(r), typeof(c))
        zeroH = norm(H,Inf) ≈ 0
		new{n,m,T}(Q, R, H, q, r, c, zeroH, terminal)
	end
end

function QuadraticViewCost(G::SparseMatrixCSC, g::Vector,
		cost::TrajOptCore.QuadraticCostFunction, k::Int)
	n,m = state_dim(cost), control_dim(cost)
	ix = (k-1)*(n+m) .+ (1:n)
	iu = ((k-1)*(n+m) + n) .+ (1:m)
	NN = length(g)

	Q = view(G,ix,ix)
	q = view(g,ix)

	if cost.Q isa Diagonal
		for i = 1:n; Q[i,i] = cost.Q[i,i] end
	else
		Q .= cost.Q
	end
	q .= cost.q

	# Point the control-dependent values to null matrices at the terminal time step
	if cost.terminal &&  NN == k*n + (k-1)*m
		R = view(spzeros(m,m), 1:m, 1:m)
		H = view(spzeros(m,n), 1:m, 1:n)
		r = view(zeros(m), 1:m)
	else
		R = view(G,iu,iu)
		H = view(G,iu,ix)
		r = view(g,iu)
		if cost.R isa Diagonal
			for i = 1:m; R[i,i] = cost.R[i,i] end
		else
			R .= cost.R
		end
		r .= cost.r
		if !TrajOptCore.is_blockdiag(cost)
			H .= cost.H
		end
	end

	QuadraticViewCost(Q, R, H, q, r, cost.c, checks=false, terminal=cost.terminal)
end

TrajOptCore.is_blockdiag(cost::QuadraticViewCost) = cost.zeroH

struct SparseSolver{n̄,n,m,T} <: ConstrainedSolver{T}
    model::AbstractModel
    obj::Objective
    E::Objective{QuadraticViewCost{n,m,T}}
    J::Objective{QuadraticViewCost{n̄,m,T}}
	J2::Objective
    Jinv::Vector{<:InvertedQuadratic{n̄,m,T,S}} where S
	conSet::SparseConstraintSet{T}
	Dblocks::Vector{SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}}

    G::SparseMatrixCSC{T,Int}               # Cost Hessian
	Ginv::SparseMatrixCSC{T,Int}            # Inverted Cost Hessian
	Gblocks::Vector{SubArray{T,2,SparseMatrixCSC{T,Int},Tuple{UnitRange{Int},UnitRange{Int}},false}}
    g::Vector{T}                            # Cost gradient
    S::Symmetric{T,SparseMatrixCSC{T,Int}}  # Shur compliment D*(H\D')
    r::Vector{T}                            # Shur compliment residual
    λ::Vector{T}                            # Lagrange multipliers
    δZ::Primals{n,m,T}                      # primal step
	Z::Primals{n,m,T}                       # primals
    Z̄::Primals{n,m,T}                       # primals (temp)

	# Line Search
	merit::TrajectoryOptimization.MeritFunction
	crit::TrajectoryOptimization.LineSearchCriteria
	ls::TrajectoryOptimization.LineSearch

    DH::Transpose{T,SparseMatrixCSC{T,Int}} # Partial Shur compliment (D/H)

end

function SparseSolver(prob::Problem{<:Any,T}) where T
	n,m,N = size(prob)
	n̄ = RobotDynamics.state_diff_size(prob.model)
	NN = N*n + (N-1)*m
	p = num_constraints(prob)
	P = sum(p)
	zinds = [(k-1)*(n+m) .+ (1:n+m) for k = 1:N-1]
	push!(zinds, (N-1)*(n+m) .+ (1:n))

	# Constraints
	conSet = SparseConstraintSet(prob.model, prob.constraints)
	Dblocks = map(1:N) do k
		off = sum(p[1:(k-1)]) .- n*(k>1)
		len = p[k] + n*(k>1)
		view(conSet.D, off .+ (1:len), zinds[k])
	end

	# Objective
	G = spzeros(NN,NN)
	Ginv = spzeros(NN,NN)
	Gblocks = [view(Ginv,zinds[k],zinds[k]) for k = 1:N]
	g = zeros(NN)
	J = Objective([LQR.QuadraticViewCost(
			G, g, QuadraticCost{Float64}(n, m, terminal=(k==N)),k)
			for k = 1:N])
	J2 = TrajOptCore.build_cost_expansion(prob.obj, prob.model)[1]
	if prob.model isa LieGroupModel
		E = QuadraticObjective(n,m,N)
	else
		E = J
	end
    Jinv = InvertedQuadratic.(J.cost)      # inverted cost  (error state)

	# Initialze other matrices
	S = Symmetric(spzeros(P,P))
	r = zeros(P)
	λ = zeros(P)

	# Primals
	δZ = Primals(prob)
	Z = Primals(prob)
	Z̄ = Primals(prob)
	copyto!(Z, prob.Z)

	iz = 1:n+m
    _Z = [ViewKnotPoint(view(Z.Z, iz .+ (k-1)*(n+m)), n, m, prob.Z[k].dt, prob.Z[k].t) for k = 1:N-1]
    push!(_Z, ViewKnotPoint(view(Z.Z, NN-n+1:NN), n, m, 0.0, prob.tf))

	DH = transpose(spzeros(NN,P))
	for k = 1:N
		_Z[k].z .= get_z(prob.Z[k])
	end

	# Line Search
	merit = TrajectoryOptimization.L1Merit()
	ls = TrajectoryOptimization.SecondOrderCorrector()
	crit = TrajectoryOptimization.WolfeConditions()

	SparseSolver(prob.model, prob.obj, E, J, J2, Jinv, conSet, Dblocks,
		G, Ginv, Gblocks, g, S, r, λ, δZ, Z, Z̄, merit, crit, ls, DH)
end

function reset!(solver::SparseSolver)
	solver.merit.μ = 1.0
	solver.λ .*= 0
	solver.δZ.Z .*= 0
end

@inline TrajOptCore.get_objective(solver::SparseSolver) = solver.obj
@inline TO.get_cost_expansion(solver::SparseSolver) = solver.J
@inline TO.get_solution(solver::SparseSolver) = solver.Z  # current estimate
@inline TO.get_step(solver::SparseSolver) = solver.δZ
@inline TrajOptCore.get_constraints(solver::SparseSolver) = solver.conSet

@inline TO.get_primals(solver::SparseSolver) = solver.Z̄   # z + α⋅dz
function TO.get_primals(solver::SparseSolver, α)
	Z̄ = vect(solver.Z̄)
	Z = vect(solver.Z)
	dZ = vect(solver.δZ)

	Z̄ .= Z
	BLAS.axpy!(α, dZ, Z̄)
	return solver.Z̄
end

@inline TrajOptCore.cost(solver::SparseSolver, Z::Primals) = cost(solver, traj(Z))
@inline TrajOptCore.evaluate!(conSet::TrajOptCore.AbstractConstraintSet, Z::Primals) = evaluate!(conSet, traj(Z))

function update!(solver::SparseSolver, Z = solver.Z)
	Z = Z.Z_
	evaluate!(solver.conSet, Z)
	jacobian!(solver.conSet, Z)
	cost_expansion!(solver.E, solver.obj, Z)
	cost_expansion!(solver.J2, solver.obj, Z)
	update_cholesky!(solver.Jinv, solver.J)
end


function calc_Ginv!(solver::SparseSolver)
	calc_Ginv!(solver.Gblocks, solver.J2)
end

function calc_Ginv!(Gblocks, obj::Objective)
	for k in eachindex(Gblocks)
		TrajOptCore.invert!(Gblocks[k], obj[k])
	end
end

function calc_DH!(solver::SparseSolver)
	for k in eachindex(solver.Dblocks)
		inds = solver.Dblocks[k].indices
		solver.DH.parent[inds[2],inds[1]] .= solver.Jinv[k].chol\Matrix(solver.Dblocks[k])'
		solver.δZ[inds[2]] .= solver.Jinv[k].chol\solver.g[inds[2]]
	end
end

function _solve!(solver::SparseSolver)
	H,g = solver.G, solver.g
	D,d = solver.conSet.D, solver.conSet.d
	Z,λ = solver.Z.Z, solver.λ
	P,NN = size(D)
	# K = [H D'; D zeros(P,P)]
	# r = -[g; d]
	# return K\r

	LQR.calc_Ginv!(solver)
	HD = solver.Ginv*D'
	Hg = solver.Ginv*g

    # HD = Symmetric(H)\D'
	# Hg = Symmetric(H)\g
	# calc_DH!(solver)
	# HD = solver.DH.parent
	# Hg = solver.δZ
	S = D*HD
    # r = D*Hg - d
	r = d - D*Hg
	# r = -d
    λ = Symmetric(S)\r
	solver.λ .= λ
    solver.δZ.Z .= -HD*λ - Hg
end

function step!(solver::SparseSolver)
	merit, ls, crit = solver.merit, solver.ls, solver.crit

	# Update the cost and constraint expansions
	update!(solver)

	# Check convergence criteria
	feas_p = max_violation(solver, recalculate=false)
	feas_d = residual(solver, recalculate=false)
	ϵ_d = 1e-5
	ϵ_p = 1e-5
	@show feas_p
	@show feas_d
	if feas_p < ϵ_p && feas_d < ϵ_d
		return true
	end

	# Update merit penalty
	TO.update_penalty!(merit, solver)

	# Solve the QOCP (Quadratic Optimal Control Problem)
	_solve!(solver)

	# Run the line search
	α = TO.line_search(ls, crit, merit, solver)
	@show α

	# Save the new iterate
	copyto!(solver.Z.Z, solver.Z̄.Z)

	return false
end

function solve!(solver::SparseSolver)
	reset!(solver)
	iters = 10
	update!(solver)

	for i = 1:iters
		converged = step!(solver)
		if converged
			break
		end
	end
end

function residual(solver::SparseSolver; recalculate=true)
	if recalculate
		conSet = get_constraints(solver)
		Z = get_trajectory(solver)
		jacobian!(conSet, Z)
		cost_gradient!(TrajOptCore.get_cost_expansion(solver), get_objective(solver), Z)
	end
	D = solver.conSet.D
	g = solver.g
	λ = solver.λ
	feas_d = norm(g + D'λ)
end

function TO.cost_dgrad(solver::SparseSolver, Z=TrajOptCore.get_primals(solver),
		dZ=TrajOptCore.get_step(solver); recalculate=true)
	if recalculate
		E = TO.get_cost_expansion(solver)
		obj = get_objective(solver)
		cost_gradient!(E, obj, Z.Z_)
	end
	solver.g'dZ.Z
end

function TO.norm_dgrad(solver::SparseSolver, Z=TrajOptCore.get_primals(solver),
		dZ=TrajOptCore.get_step(solver); recalculate=true, p=1)
    conSet = get_constraints(solver)
	if recalculate
		Z_ = Z.Z_
		evaluate!(conSet, Z_)
		jacobian!(conSet, Z_)
	end
	D,d = solver.conSet.D, solver.conSet.d
	TrajOptCore.norm_dgrad(d, D*dZ.Z, p)
end

function cost_dhess(solver::SparseSolver, Z=TrajOptCore.get_primals(solver),
		dZ=TrajOptCore.get_step(solver); recalculate=true)
	E = TrajOptCore.get_cost_expansion_error(solver)
	if recalculate
		obj = get_objective(solver)
		cost_hessian!(E, obj, Traj(Z))
	end
	dot(dZ.Z, solver.G, dZ.Z)
end

function TO.second_order_correction!(solver::SparseSolver)
	Z = TO.get_primals(solver)     # get the current value of z + α⋅δz
	D = solver.conSet.D
	d = solver.conSet.d
	G = solver.G
	g = solver.g
	λ = solver.λ
	P,NN = size(D)
	evaluate!(get_constraints(solver), Z.Z_)  # update constraints at current step

	# jacobian!(get_constraints(solver), get_trajectory(solver))
	# F = cholesky!(D*D')

	δx̂ = -D'*((D*D')\d)
	# return δx̂
	Z.Z .+= δx̂
end

function project!(solver::SparseSolver)
	conSet = get_constraints(solver)
	Z = TrajOptCore.get_primals(solver)
	D,d = conSet.D, conSet.d
	G,g = solver.G, solver.g
	# GinvD = (G\D')
	# S = D*GinvD
	S = D*D'
	F = cholesky(Symmetric(S))

	evaluate!(conSet, Z.Z_)
	c_max0 = max_violation(conSet)
	α = 1.0
	for i = 1:10
		λ = -(F\d)
		dZ = D'λ
		Z.Z .+= α*dZ
		evaluate!(conSet, Z.Z_)
		c_max = max_violation(conSet)
		@show c_max
		if c_max < c_max0
			return c_max
		end
		α /= 2.0
	end
end
