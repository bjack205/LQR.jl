import TrajectoryOptimization: QuadraticExpansion

export
    CholeskySolver

struct MutableKnotPoint{T,N,M,NM} <: AbstractKnotPoint{T,N,M}
    z::SizedVector{NM,T,1}
    _x::SubArray{T,1,Vector{T}}
    _u::SubArray{T,1,Vector{T}}
    dt::T
    t::T
    function MutableKnotPoint(n::Int, m::Int, z::MVector,
            dt::Real, t::Real)
		@assert length(z) == n+m
		@assert dt >= 0
		dt,t = promote(dt,t)
		z = SizedVector{n+m}(z)
		_x = view(z.data, 1:n)
		_u = view(z.data, n .+ (1:m))
        new{typeof(dt),n,m,n+m}(z, _x, _u, dt, t)
    end
end

function MutableKnotPoint(x, u, dt, t=0.0)
	n,m = length(x), length(u)
	MVector{n+m}([x;u])
    ViewKnotPoint(n, m, z, dt, t)
end

@inline RobotDynamics.state(z::MutableKnotPoint) = z._x
@inline RobotDynamics.control(z::MutableKnotPoint) = z._u

function Base.:*(a::Real, z::MutableKnotPoint{<:Any,n,m}) where {n,m}
	StaticKnotPoint(z.z*a, SVector{n}(1:n), SVector{m}(n .+ (1:m)), z.dt, z.t)
end

import RobotDynamics.get_z

struct CholeskySolver{n̄,n,m,n̄m,nm,T} #<: ConstrainedSolver{T}
    model::AbstractModel
    obj::Objective
    E::QuadraticExpansion{n,m,T}
    J::QuadraticExpansion{n̄,m,T}
    Jinv::Vector{<:InvertedQuadratic{n̄,m,T,S}} where S
    conSet::BlockConstraintSet{T}
    shur_blocks::Vector{<:BlockTriangular3{<:Any,<:Any,<:Any,T}}
    chol_blocks::Vector{<:BlockTriangular3{<:Any,<:Any,<:Any,T}}
    δZ::Vector{MutableKnotPoint{T,n̄,m,n̄m}}
    Z::Vector{KnotPoint{T,n,m,nm}}
    Z̄::Vector{StaticKnotPoint{T,n,m,nm}}
    G::Vector{SizedMatrix{n,n̄,T,2}}
	res::Vector{MVector{n̄m,T}}  # residual

	# Line Search
	merit::MeritFunction
	crit::LineSearchCriteria
	ls::LineSearch
end

function CholeskySolver(prob::Problem)
    n̄ = RobotDynamics.state_diff_size(prob.model)
    n,m,N = size(prob)
	J,E = TO.build_cost_expansion(prob.obj, prob.model)
    Jinv = InvertedQuadratic.(J.cost)      # inverted cost  (error state)

    conSet = BlockConstraintSet(prob.model, get_constraints(prob))
    shur_blocks = build_shur_factors(conSet, :U)
    chol_blocks = build_shur_factors(conSet, :U)

    dt = prob.tf / (N - 1)
    δZ = [MutableKnotPoint(n̄,m, (@MVector zeros(n̄+m)), z.dt, z.t) for z in prob.Z]
    Z = deepcopy(prob.Z)
    Z̄ = [StaticKnotPoint(z) for z in prob.Z]

	G = [SizedMatrix{n,n̄}(zeros(n,n̄)) for k = 1:N+1]  # add one to the end to use as an intermediate result

	res = [@MVector zeros(n̄+m) for k = 1:N]

	# Line Search
	merit = TO.L1Merit()
	crit = TO.WolfeConditions()
	ls = TO.SecondOrderCorrector()

    CholeskySolver(prob.model, prob.obj, E, J, Jinv, conSet, shur_blocks, chol_blocks,
		δZ, Z, Z̄, G, res, merit, crit, ls)
end

function reset!(solver::CholeskySolver)
	solver.merit.μ = 1.0
	N = size(solver)[3]
	for k = 1:N
		solver.δZ[k].z .*= 0
	end
end

@inline TO.get_objective(solver::CholeskySolver) = solver.obj
@inline get_cost_expansion(solver::CholeskySolver) = solver.J
@inline get_solution(solver::CholeskySolver) = solver.Z
@inline get_step(solver::CholeskySolver) = solver.δZ
@inline get_primals(solver::CholeskySolver) = solver.Z̄
@inline TO.get_constraints(solver::CholeskySolver) = solver.conSet
@inline TO.get_trajectory(solver::CholeskySolver) = solver.Z

num_vars(solver::CholeskySolver{n,<:Any,m}) where {n,m} = length(solver.obj)*n + (length(solver.obj)-1)*m
Base.size(solver::CholeskySolver{<:Any,n,m}) where {n,m} = n,m,length(solver.obj)

TO.get_model(solver::CholeskySolver) = solver.model

function solve!(solver::CholeskySolver)
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

function step!(solver::CholeskySolver)
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
	copyto!(solver.Z, solver.Z̄)

	return false
end

function update!(solver::CholeskySolver)
    Z = solver.Z
	RobotDynamics.state_diff_jacobian!(solver.G, solver.model, Z)
    evaluate!(solver.conSet, Z)
    jacobian!(solver.conSet, Z)
    error_expansion!(solver.conSet, solver.model, solver.G)
	cost_expansion!(solver.E, solver.obj, Z)
	error_expansion!(solver.J, solver.E, solver.model, Z, solver.G)
	update_cholesky!(solver.Jinv, solver.J)
end

function _solve!(solver::CholeskySolver)
	chol = solver.chol_blocks
	shur = solver.shur_blocks

	# Calculate S = D*(H\D) and r = D*(H\g) -d
    calculate_shur_factors!(shur, solver.Jinv, solver.conSet.blocks)

	# Calculate cholesky factorization of S
    cholesky!(chol, shur)

	# Solve λ = -S\r
    forward_substitution!(chol)
    backward_substitution!(chol)

	# Calculate z = -H\(D'λ + g)
    calculate_primals!(solver.δZ, solver.Jinv, chol, solver.conSet.blocks)
end


function calculate_primals!(dZ::Traj, Jinv::Vector{<:InvertedQuadratic}, chol, blocks)
    N = length(blocks)
	calc_residual!(blocks, Jinv, chol)

	for k = 1:N
		z = dZ[k]
		calc_primals!(get_z(z), Jinv[k], blocks[k])
	end
end

function calc_primals!(z, Jinv::InvertedQuadratic, block)
	z .= block.res
	ldiv!(Jinv.chol, z)
	z .*= -1
end

function calc_residual!(blocks::Vector, Jinv::Vector{<:InvertedQuadratic}, chol, Ginv::Bool=true)
	N = length(blocks)
    for k = 1:N
        g = gradient(Jinv[k])
        if k == 1
            calc_residual!(blocks[k], Jinv[k], chol[k], Ginv)
            # sol.Z_[k].z .+= -Hinv*(blocks[k].C'chol[k].μ .+ blocks[k].D1'chol[k].λ)
        else
            calc_residual!(blocks[k], Jinv[k], chol[k], chol[k-1], Ginv)
            # sol.Z_[k].z .+= -Hinv*(blocks[k].D2'chol[k-1].λ .+ blocks[k].C'chol[k].μ
            #     .+ blocks[k].D1'chol[k].λ)
        end
    end
end

function calc_residual!(block, Jinv, chol, Ginv::Bool=true)
    # first time step
	z = block.res
    mul!(z, block.D1', chol.λ)
    mul!(z, block.C', chol.μ, 1.0, 1.0)
	if Ginv
		add_gradient!(z, Jinv)
	end
    # ldiv!(Jinv.chol, z)
end

function calc_residual!(block, Jinv, chol, chol_prev, Ginv::Bool=true)
	z = block.res
    mul!(z, block.D1', chol.λ)
    mul!(z, block.C', chol.μ, 1.0, 1.0)
    mul!(z, block.D2', chol_prev.λ, 1.0, 1.0)
	if Ginv
		add_gradient!(z, Jinv)
	end
    # ldiv!(Jinv.chol, z)
end

function residual(solver::CholeskySolver; recalculate=true)
	if recalculate
		conSet = get_constraints(solver)
		Z = get_trajectory(solver)
		jacobian!(conSet, Z)
		cost_gradient!(TO.get_cost_expansion(solver), get_objective(solver), Z)
		update_cholesky!(solver.Jinv, solver.J)
	end
	calc_residual!(solver.conSet.blocks, solver.Jinv, solver.chol_blocks)
	res = zeros(length(solver.obj))
	for (k,block) in enumerate(solver.conSet.blocks)
		res[k] = norm(block.res)
	end
	norm(res)
end

function second_order_correction!(solver::CholeskySolver{<:Any,<:Any,m}) where m
	# Calculate dZ = -D*(D*D')\d
	Z = TO.get_primals(solver)      # current value of z + α*δz
	evaluate!(solver.conSet, Z)  # update constraints

	calculate_shur_factors!(solver.shur_blocks, solver.Jinv, solver.conSet.blocks, false)
	cholesky!(solver.chol_blocks, solver.shur_blocks)
	forward_substitution!(solver.chol_blocks)
	backward_substitution!(solver.chol_blocks)
	calc_residual!(solver.conSet.blocks, solver.Jinv, solver.chol_blocks, false)
	N = size(solver)[3]
	for k = 1:N-1
		solver.conSet.blocks[k].res .*= -1
		Z[k] += solver.conSet.blocks[k].res
	end
	solver.conSet.blocks[N].res .*= -1
	Z[N] = StaticKnotPoint(Z[N],
		[state(Z[N]) + solver.conSet.blocks[N].res; @SVector zeros(m)])
	Z
end


# ~~~~~~~~ Get Block Pieces ~~~~~~~~~~~ #

function get_linearized_constraints(solver::CholeskySolver)
	P = sum(solver.conSet.p)
	NN = num_vars(solver)
	D = spzeros(P, NN)
	d = zeros(P)
	LQR.copy_blocks!(D, d, solver.conSet.blocks)
	return D,d
end

function get_cost_expansion(solver::CholeskySolver)
	n,m,N = size(solver)
	NN = num_vars(solver)
	H = spzeros(NN,NN)
	g = zeros(NN)
	build_H!(H, solver.J)
	ix = 1:n
	iu = n .+ (1:m)
	for k = 1:N-1
		g[ix] .= solver.J[k].q
		g[iu] .= solver.J[k].r
		ix = (n+m) .+ (ix)
		iu = (n+m) .+ (iu)
	end
	g[ix] .= solver.J[N].q
	return H,g
end

function get_step(solver::CholeskySolver)
	n,m,N = size(solver)
	NN = num_vars(solver)
	Z = zeros(NN)
	ix = 1:n
	iu = n .+ (1:m)
	for k = 1:N-1
		Z[ix] .= state(solver.δZ[k])
		Z[iu] .= control(solver.δZ[k])
		ix = (n+m) .+ ix
		iu = (n+m) .+ iu
	end
	Z[ix] .= get_z(solver.δZ[N])
	return Z
end

function get_residual(solver::CholeskySolver)
	n,m,N = size(solver)
	NN = num_vars(solver)
	Z = zeros(NN)
	ix = 1:n
	iu = n .+ (1:m)
	blocks = solver.conSet.blocks
	for k = 1:N-1
		Z[ix] .= blocks[k].res[1:n]
		Z[iu] .= blocks[k].res[n .+ (1:m)]
		ix = (n+m) .+ ix
		iu = (n+m) .+ iu
	end
	Z[ix] .= blocks[N].res[1:n]
	return Z
end

function get_shur_factors(solver::CholeskySolver)
	P = sum(solver.conSet.p)
	S = zeros(P,P)
	d_ = zeros(P)
	λ = zeros(P)
	copy_shur_factors!(S, d_, λ, solver.shur_blocks)
	return Symmetric(S, :U), d_, λ
end

function get_multipliers(solver::CholeskySolver)
	P = sum(solver.conSet.p)
	U = zeros(P,P)  # upper cholesky
	d_ = zeros(P)
	λ = zeros(P)
	copy_shur_factors!(U, d_, λ, solver.chol_blocks)
	return λ
end

function get_cholesky(solver::CholeskySolver)
	P = sum(solver.conSet.p)
	U = zeros(P,P)  # upper cholesky
	d_ = zeros(P)
	λ = zeros(P)
	copy_shur_factors!(U, d_, λ, solver.chol_blocks)
	return UpperTriangular(U)
end

function norm_residual(solver::CholeskySolver{n,<:Any,m}) where {n,m}
	res = solver.res
	ix = 1:n
	iu = n .+ (1:m)

	for (k,cost) in enumerate(TO.get_cost_expansion(solver))
		res[k] = [cost.q; cost.r]
	end
end
