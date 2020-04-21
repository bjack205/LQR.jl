using MeshCat
using TrajOptPlots
using TrajectoryOptimization
using RobotZoo
using ForwardDiff
using TrajOptCore
using LQR
using StaticArrays
using LinearAlgebra
using RobotDynamics
const TO = TrajectoryOptimization


if !isdefined(Main, :vis)
    vis = Visualizer()
    open(vis)
    set_mesh!(vis, RobotZoo.DubinsCar())
end

prob, = Problems.DubinsCar(:turn90, N=11)
ilqr = iLQRSolver(prob)
TO.solve!(ilqr)
visualize!(vis, ilqr)

prob, = Problems.DubinsCar(:turn90, N=11)
TrajOptCore.add_dynamics_constraints!(prob)
n,m,N = size(prob)
NN = LQR.num_vars(prob)
P = sum(num_constraints(prob))
Z0 = LQR.Primals(prob).Z

zinds = [(k-1)*(n+m) .+ (1:n+m) for k = 1:N]
zinds[end] = (N-1)*(n+m) .+ (1:n)
_zinds = [SVector{length(ind)}(ind) for ind in zinds]

# objective
function f(Z)
	_Z = [StaticKnotPoint(prob.Z[k], Z[_zinds[k]]) for k = 1:N]
	J = zeros(eltype(Z), N)
	TrajOptCore.cost!(prob.obj, _Z, J)
	return sum(J)
end
∇f(Z) = ForwardDiff.gradient(f, Z)
∇²f(Z) = ForwardDiff.hessian(f, Z)

# constraints
function c(Z)
	_Z = [StaticKnotPoint(prob.Z[k], Z[_zinds[k]]) for k = 1:N]
	val = state(_Z[1]) - prob.x0
	for k = 1:N-1
		dyn = discrete_dynamics(RK3, prob.model, _Z[k]) - state(_Z[k+1])
		val = [val; dyn]
	end
	return val
end
∇c(Z) = ForwardDiff.jacobian(c, Z)

# merit function
μ = 1.0
ϕ(x::AbstractVector) = f(x) + μ*norm(c(x),1)
ϕ′(x, dx) = ∇f(x)'dx - μ*norm(c(x), 1)

# Define KKT system
K(x) = begin
	K1 = [∇²f(x) ∇c(x)']
	K2 = [∇c(x) @SMatrix zeros(P,P)]
	[K1; K2]
end
r(x) = -[∇f(x) + ∇c(x)'λ; c(x)]
r2(x) = -[∇f(x); c(x)]


# Line search
function line_search(ϕ, ϕ′, x, dx)
    α = 1.0
	η = 1e-4
    ρ = 0.5
    for i = 1:10
        if ϕ(x + α*dx) ≤ ϕ(x) + η*α*ϕ′(x, dx)
			println("α = $α")
			return x + α*dx
		elseif α == 1
			println("trying second-order correction")
			A = ∇c(x)
			dx̂ = -A'*((A*A')\c(x + dx))
			if ϕ(x + dx + dx̂) < ϕ(x) + η*ϕ′(x, dx)
				println("success")
				return x + dx + dx̂
			else
				α *= ρ
			end
		else
			α *= ρ
        end
    end
	@warn "Line Search Failed"
end

# Initialize
Z = LQR.Primals(prob)
x = Z.Z
λ = zeros(P)
dx = zero(x)
visualize!(vis, prob.model, Z.Z_)

dy = K(x)\r(x)
dx = dy[1:NN]
dλ = dy[NN+1:end]
norm(c(x),Inf)
x = line_search(ϕ, ϕ′, x, dx)
ϕ(x)
Z.Z .= x
visualize!(vis, prob.model, Z.Z_)

μ = 1
alphas = range(-3,3,length=21)
plot(alphas, [ϕ(x + α*dx) for α in alphas])
plot!(alphas, [f(x + α*dx) for α in alphas])
plot!(alphas, [norm(c(x + α*dx),1) for α in alphas])

ϕ(x)
norm(c(x), Inf)
