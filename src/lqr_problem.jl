using TrajOptCore
using RobotDynamics
using StaticArrays
using LinearAlgebra
using RobotZoo
using BenchmarkTools

struct LQRProblem{T,N,M,TQ,TR}
    Qf::TQ
    Q::TQ
    R::TR
    A::SizedMatrix{N,N,T,2}
    B::SizedMatrix{N,M,T,2}
    x0::SVector{N,T}
    u0::SVector{M,T}
    dt::T
    N::Int
end

function LQRProblem(model::AbstractModel, Q, R, Qf, z0::AbstractKnotPoint, N::Int; integration=RK3)
    A,B = linearize(integration, model, z0)
    return LQRProblem(Qf, Q, R, A, B, state(z0), control(z0), z0.dt, N)
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


function build_least_squares!(T::AbstractMatrix, L::AbstractMatrix, prob::LQRProblem{<:Any,n,m}) where {n,m}
    A,B = prob.A, prob.B
    ix,iu = 1:n, 1:m
    N = prob.N
    tmpA = SizedMatrix{n,n}(Diagonal(@SVector ones(n)))
    tmpB = copy(B)
    for j = 0:N-1          # loop powers of A
        ix_ = ix .+ j*n
        L[ix_,ix] .= tmpA
        for i = 1:(N-1-j)       # loop over columns
            ix_ = ix .+ (i+0)*n .+ j*n
            iu_ = iu .+ (i-1)*m
            # @show ix_, iu_
            mul!(tmpB, tmpA, B)
            T[ix_, iu_] .= tmpB
        end
        mul!(tmpA, A, tmpA)
    end
end
T = zeros(N*n, (N-1)*m)
L = zeros(N*n, n)
prob.B .= 1
prob.A .= 1.1*I(3)
build_least_squares!(T, L, prob)
@btime build_least_squares!($T, $L, $prob)

model = RobotZoo.DubinsCar()
n,m = size(model)
N,dt = 101, 0.01
x0,u0 = zeros(model)
z0 = KnotPoint(x0,u0,dt)

Q = Diagonal(@SVector fill(1.,n))
R = Diagonal(@SVector fill(1.,m))
prob = LQRProblem(model, Q, R, 10Q, z0, 101)
