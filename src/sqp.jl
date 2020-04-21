using LinearAlgebra
using StaticArrays
using ForwardDiff
using Plots

Q = Diagonal(SA[1,2])
q = SA[1,1]
f(x) = 0.5*x'Q*x + q'x
c(x) = SA[x[1]^2 + x[2]]

x = @SVector rand(2)
∇f(x) = Q*x + q
∇f(x) ≈ ForwardDiff.gradient(f, x)

∇²f(x) = Q
∇²f(x) ≈ ForwardDiff.hessian(f, x)

∇c(x) = SA[2x[1] 1]
∇c(x) ≈ ForwardDiff.jacobian(c, x)

function plot_prob(x)
    xs = ys = range(-1,1, length=101)
    contour(xs,ys, (x,y) -> f(SA[x,y]))
    plot!(xs, -xs.^2, label="c(x)", lw=2, c=:red)
    scatter!([x[1],], [x[2],], label="", c=:black, markersize=5)
end

# Find the solution (mostly) analytically
g(x) = 2Q[2,2]*x^3 + (Q[1,1] - 2*q[2])*x + q[1]
dg(x) = 6Q[2,2]*x^2 + (Q[1,1] - 2*q[2])
function newton_root(g, dg, x)
	x_prev = x
	for i = 1:100
		x = x_prev - g(x)/dg(x)
		if abs(x - x_prev) < 1e-10
			break
		end
		x_prev = x
	end
	return x
end
x1 = newton_root(g, dg, 0)
x2 = -x1^2
xstar = SA[x1, x2]
λstar = SA[-(Q[2,2]*x2 + q[2])]
c(xstar)[1]
f(xstar)
norm(∇f(xstar) + ∇c(xstar)'λstar)

x_uncon = -Q\q
plot_prob(x_uncon)
plot_prob(xstar)

# merit function
μ = 1.0
ϕ(x::AbstractVector) = f(x) + μ*norm(c(x),1)
ϕ′(x, dx) = ∇f(x)'dx - μ*norm(c(x), 1)
ϕ(xstar)


# Define KKT system
K(x) = begin
	K1 = [SMatrix{2,2}(∇²f(x)) ∇c(x)']
	K2 = [∇c(x) SA[0.]]
	[K1; K2]
end
r(x) = -[∇f(x) + ∇c(x)'λ; c(x)]

r2(x) = -[∇f(x); c(x)]

# Line search
function line_search(ϕ, ϕ′, x, dx, λ, dλ)
    α = 1.0
	η = 1e-4
    ρ = 0.5
    for i = 1:10
        if ϕ(x + α*dx) ≤ ϕ(x) + η*α*ϕ′(x, dx)
			println("α = $α")
			return x + α*dx, λ + α*dλ
		elseif α == 1
			println("trying second-order correction")
			dx̂ = -A'*((A*A')\c(x + dx))
			if ϕ(x + dx + dx̂) < ϕ(x) + η*ϕ′(x, dx)
				println("success")
				return x + dx + dx̂, λ + dλ
			else
				α *= ρ
			end
		else
			α *= ρ
        end
    end
	@warn "Line Search Failed"
end
ϕ(α::Real) = ϕ(x + α*dx)
ϕ′(α::Real) = ∇f(x + α*dx)'dx - μ*norm(c(x + α*dx),1)

# Initialize
x0 = SA[0., 0.]
x = x0
λ = SA[0.]
dx = zero(x)
plot_prob(x0)

# Only use ∇f in rhs
norm(c(x),Inf)
dy = K(x)\r2(x)
dx = dy[1:2]
dλ = dy[3:3] - λ
x,λ = line_search(ϕ, ϕ′, x, dx, λ, dλ)
plot_prob(x)
norm(λstar- λ)
norm(xstar - x)

# Newton step (these two are both equivalent)
norm(c(x),Inf)
dy = K(x)\r(x)
dx = dy[1:2]
dλ = dy[3:3]
x,λ = line_search(ϕ, ϕ′, x, dx, λ, dλ)
plot_prob(x)
norm(λstar- λ)
norm(xstar - x)

d = c(x + dx) - ∇c(x)*dx
r_ = -[∇f(x); d]
dŷ = K(x)\r_
dx̂ = dŷ[1:2]
A = ∇c(x)
dx̂ = -A'*((A*A')\c(x + dx))


norm(x - xstar)

μ = 1
alphas = range(-3,3,length=21)
plot(alphas, ϕ.(alphas))
plot!(alphas, [f(x + α*dx) for α in alphas])
plot!(alphas, [norm(c(x + α*dx),1) for α in alphas])

ϕ(x)
norm(c(x), Inf)
