struct LQRProblem{n,m,T,TQ,TR}
    Qf::TQ
    Q::TQ
    R::TR
    A::SizedMatrix{n,n,T,2}
    B::SizedMatrix{n,m,T,2}
    x0::SVector{n,T}
    u0::SVector{m,T}
    dt::T
    N::Int
end

function LQRProblem(model::AbstractModel, Q, R, Qf, z0::AbstractKnotPoint, N::Int; integration=RK3)
    A,B = linearize(integration, model, z0)
    return LQRProblem(Qf, Q, R, A, B, state(z0), control(z0), z0.dt, N)
end

Base.size(prob::LQRProblem{n,m}) where {n,m} = n,m,prob.N
function num_vars(prob::LQRProblem)
    n,m,N = size(prob)
    return N*n + (N-1)*m
end

struct ViewKnotPoint{T,N,M,NM} <: AbstractKnotPoint{T,N,M,NM}
    z::SubArray{T,1,Vector{T},Tuple{UnitRange{Int}},true}
    _x::SVector{N,Int}
    _u::SVector{M,Int}
    dt::T
    t::T
    function ViewKnotPoint(z::SubArray, _x::SVector{N,Int}, _u::SVector{M,Int},
            dt::T1, t::T2) where {N,M,T1,T2}
        T = promote_type(T1,T2)
        @assert length(z) == N+M
        new{T,N,M,N+M}(z, _x, _u, T(dt), T(t))
    end
end
