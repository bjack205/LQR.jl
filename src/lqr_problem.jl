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

struct ViewKnotPoint{T,N,M,NM} <: AbstractKnotPoint{T,N,M}
    z::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::T
    t::T
    function ViewKnotPoint(z::SubArray, _x::SVector{N,Int}, _u::SVector{M,Int},
            dt::T1, t::T2) where {N,M,T1,T2}
        T = promote_type(T1,T2)
        new{T,N,M,N+M}(z, _x, _u, T(dt), T(t))
    end
end

function ViewKnotPoint(z::SubArray, n, m, dt, t=0.0)
    ix = SVector{n}(1:n)
    iu = SVector{m}(n .+ (1:m))
    ViewKnotPoint(z, ix, iu, dt, t)
end

struct LQRSolution{n,m,T,nm}
    Z::Vector{T}
    Z_::Vector{ViewKnotPoint{T,n,m,nm}}
    X::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}
    U::Vector{SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}}
    X_::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    U_::SubArray{T,1,Vector{T},Tuple{Vector{Int}},false}
    K::Vector{SizedMatrix{m,n,T,2}}
end

function LQRSolution(n,m,N,tf)
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
    K = [SizedMatrix{m,n}(zeros(m,n)) for k = 1:N-1]
    LQRSolution(Z,Z_,X,U,X_,U_,K)
end

@inline LQRSolution(prob::LQRProblem{n,m}) where {n,m} = LQRSolution(n,m, prob.N, prob.tf)
