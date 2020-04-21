import TrajOptCore: QuadraticExpansion

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
import RobotDynamics.get_z

struct CholeskySolver{n̄,n,m,n̄m,nm,T}
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
end

function CholeskySolver(prob::Problem)
    n̄ = RobotDynamics.state_diff_size(prob.model)
    n,m,N = size(prob)
	J,E = TrajOptCore.build_cost_expansion(prob.obj, prob.model)
    Jinv = InvertedQuadratic.(J.cost)      # inverted cost  (error state)

    conSet = BlockConstraintSet(prob.model, get_constraints(prob))
    shur_blocks = build_shur_factors(conSet, :U)
    chol_blocks = build_shur_factors(conSet, :U)

    dt = prob.tf / (N - 1)
    δZ = [MutableKnotPoint(n̄,m, (@MVector zeros(n̄+m)), z.dt, z.t) for z in prob.Z]
    Z = deepcopy(prob.Z)
    Z̄ = [StaticKnotPoint(z) for z in prob.Z]

	G = [SizedMatrix{n,n̄}(zeros(n,n̄)) for k = 1:N+1]  # add one to the end to use as an intermediate result

    CholeskySolver(prob.model, prob.obj, E, J, Jinv, conSet, shur_blocks, chol_blocks,
		δZ, Z, Z̄, G)
end

num_vars(solver::CholeskySolver{n,<:Any,m}) where {n,m} = length(solver.obj)*n + (length(solver.obj)-1)*m
Base.size(solver::CholeskySolver{<:Any,n,m}) where {n,m} = n,m,length(solver.obj)

TrajOptCore.get_model(solver::CholeskySolver) = solver.model

function solve!(sol, solver::CholeskySolver)
    # for i = 1:10
    #     # Update constraints
    #     evaluate!(blocks, solver.Z)
    #     jacobian!(blocks, solver.Z)
    #
    #     # Update cost function
    #     cost_expansion!(solver.J, solver.obj, solver.Z)
    # end
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
    for k = 1:N
        g = gradient(Jinv[k])
        z = dZ[k]
        if k == 1
            calc_primals!(get_z(z), Jinv[k], chol[k], blocks[k])
            # sol.Z_[k].z .+= -Hinv*(blocks[k].C'chol[k].μ .+ blocks[k].D1'chol[k].λ)
        else
            calc_primals!(get_z(z), Jinv[k], chol[k], chol[k-1], blocks[k])
            # sol.Z_[k].z .+= -Hinv*(blocks[k].D2'chol[k-1].λ .+ blocks[k].C'chol[k].μ
            #     .+ blocks[k].D1'chol[k].λ)
        end
        z.z .*= -1
    end
end

function calc_primals!(z, Jinv, chol, block)
    # first time step
    mul!(z, block.D1', chol.λ)
    mul!(z, block.C', chol.μ, 1.0, 1.0)
	add_gradient!(z, Jinv)
    ldiv!(Jinv.chol, z)
end

function calc_primals!(z, Jinv, chol, chol_prev, block)
    mul!(z, block.D1', chol.λ)
    mul!(z, block.C', chol.μ, 1.0, 1.0)
    mul!(z, block.D2', chol_prev.λ, 1.0, 1.0)
	add_gradient!(z, Jinv)
    ldiv!(Jinv.chol, z)
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
