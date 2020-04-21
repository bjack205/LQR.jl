struct LQRProblem{n,m,T,TQ,TR}
    Qf::TQ
    Q::TQ
    R::TR
    A::SizedMatrix{n,n,T,2}
    B::SizedMatrix{n,m,T,2}
    x0::SVector{n,T}
    u0::SVector{m,T}
    tf::T
    N::Int
end

function LQRProblem(model::AbstractModel, Q, R, Qf, z0::AbstractKnotPoint, N::Int, tf::Real;
        integration=RK3)
    dt = tf/(N-1)
    z0.dt = dt
    A,B = linearize(integration, model, z0)
    return LQRProblem(Qf, Q, R, A, B, state(z0), control(z0), tf, N)
end

Base.size(prob::LQRProblem{n,m}) where {n,m} = n,m,prob.N
function num_vars(prob::LQRProblem)
    n,m,N = size(prob)
    return N*n + (N-1)*m
end

struct ViewKnotPoint{T,N,M} <: AbstractKnotPoint{T,N,M}
    z::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::T
    t::T
    function ViewKnotPoint(z::SubArray, _x::SVector{N,Int}, _u::SVector{M,Int},
            dt::T1, t::T2) where {N,M,T1,T2}
        T = promote_type(T1,T2)
        new{T,N,M}(z, _x, _u, dt, t)
    end
end

function ViewKnotPoint(z::SubArray, n, m, dt, t=0.0)
    ix = SVector{n}(1:n)
    iu = SVector{m}(n .+ (1:m))
    ViewKnotPoint(z, ix, iu, dt, t)
end

struct Primals{n,m,T}
    Z::Vector{T}
    Z_::Vector{ViewKnotPoint{T,n,m}}
    X::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}
    U::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}
    X_::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    U_::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
end

function Primals(n,m,N,tf)
    dt = tf/(N-1)
    NN = N*n + (N-1)*m
    Z = zeros(NN)
    ix, iu, iz = 1:n, n .+ (1:m), 1:n+m
    iX = [ix .+ k*(n+m) for k = 0:N-1]
    iU = [iu .+ k*(n+m) for k = 0:N-2]

    tf = (N-1)*dt
    t = range(0,tf,length=N)
    dt = t[2]
    Z_ = [ViewKnotPoint(view(Z,iz .+ k*(n+m)), n, m, dt, t[k+1]) for k = 0:N-2]
    X = [view(Z, iX[k]) for k = 1:N]
    U = [view(Z, iU[k]) for k = 1:N-1]
    X_ = view(Z, vcat(iX...))
    U_ = view(Z, vcat(iU...))
    push!(Z_, ViewKnotPoint(X[end], n, m, 0.0, tf))
    Primals(Z,Z_,X,U,X_,U_)
end

@inline Primals(prob::Primals{n,m}) where {n,m} = Primals(n,m, prob.N, prob.tf)
@inline Primals(prob::Problem) = Primals(size(prob)..., prob.tf)

function Base.copyto!(Z0::Primals, Z::Traj)
    N = length(Z)
    for k = 1:N-1
        Z0.Z_[k].z .= Z[k].z
    end
    Z0.Z_[N].z .= state(Z[N])
end

@inline Base.copyto!(Z0::Primals, Z::Vector{<:Real}) = copyto!(Z0.Z, Z)
function Base.copy(Z::Primals{n,m}) where {n,m}
    tf = traj(Z)[end].t
    Z_ = Primals(n,m, length(traj(Z)), tf)
    copyto!(Z_, vect(Z))
    return Z_
end

function Base.:+(Z1::Primals, Z2::Primals)
    Z = copy(Z1)
    Z.Z .= Z1.Z .+ Z2.Z
    return Z
end

function Base.:*(a::Real, Z1::Primals)
    Z = copy(Z1)
    Z.Z .= a*Z1.Z
    return Z
end

@inline RobotDynamics.Traj(Z::Primals) = Z.Z_
