using LinearAlgebra
using StaticArrays
using UnsafeArrays

""" `BlockCholesky{T,S}`
Efficiently computes the cholesky factorization of a block-symmetric matrix
    `M = [A C'; C B]``

In the most general case (M is dense) the choleksy is computed on the entire matrix using LAPACK

If `block_diag == true` `C` is assumed to be zero, and the cholesky factorization is computed
    block-wise

If the underlying storage format is a Diagonal matrix (pass `diag=true` into the constructor)
    `C` is assumed to be zero and `A` and `B` are assumed to be diagonal matrices.
    In this case, calling `cholesky!(chol, A, B)` computes the inverse and does element-wise
    multiplication when solving the system using `ldiv!(chol, b)` or `chol\\b`
"""
struct BlockCholesky{T,S}
    M::S
    F::Cholesky{T,S}
    A::SubArray{T,2,S,Tuple{UnitRange{Int},UnitRange{Int}},false}
    B::SubArray{T,2,S,Tuple{UnitRange{Int},UnitRange{Int}},false}
    C::SubArray{T,2,S,Tuple{UnitRange{Int},UnitRange{Int}},false}
    Ct::SubArray{T,2,S,Tuple{UnitRange{Int},UnitRange{Int}},false}
    block_diag::Bool
    uplo::Char
end

function BlockCholesky(n::Int, m::Int; diag=false, kwargs...)
    if diag
        BlockCholesky(Diagonal(zeros(n+m)), n,m; kwargs...)
    else
        BlockCholesky(zeros(n+m,n+m), n, m; kwargs...)
    end
end

"""
[A C';
 C B]
"""
function BlockCholesky(M::S, n::Int, m::Int; block_diag=false, uplo::Char='U') where S <: AbstractMatrix
    ix = 1:n
    iu = n .+ (1:m)
    A = view(M, ix,ix)
    B = view(M, iu,iu)
    C = view(M, iu,ix)
    Ct = view(M, ix,iu)

    F = Cholesky(M, uplo, 0)
    BlockCholesky(M, F, A, B, C, Ct, block_diag, uplo)
end

"Take cholesky of the entire matrix"
function LinearAlgebra.cholesky!(chol::BlockCholesky, A, B, C)
    if chol.block_diag
        cholesky!(chol, A, B)
    else
        chol.A .= A
        chol.B .= B
        chol.C .= C
        transpose!(chol.Ct, C)
        LAPACK.potrf!(chol.uplo, chol.M)
    end
    return chol.F
end

"Take cholesky of diagonal blocks"
function LinearAlgebra.cholesky!(chol::BlockCholesky, A, B)
    chol.A .= A
    chol.B .= B
    LAPACK.potrf!(chol.uplo, chol.A)
    LAPACK.potrf!(chol.uplo, chol.B)
    return nothing
end

""" Cholesky of a diagonal block.
Actually computes the inverse to avoid divisions during the solve
"""
function LinearAlgebra.cholesky!(chol::BlockCholesky{<:Any,<:Diagonal}, A::Diagonal, B::Diagonal)
    n = size(chol.A,1)
    m = size(chol.B,1)
    for i = 1:n
        chol.M.diag[i] = inv(A.diag[i])
    end
    for i = 1:m
        chol.M.diag[n+i] = inv(B.diag[i])
    end
end

@inline LinearAlgebra.ldiv!(chol::BlockCholesky, b::AbstractVecOrMat) = ldiv!(chol.F, b)

@inline LinearAlgebra.ldiv!(chol::BlockCholesky{<:Any,<:Diagonal}, b::AbstractVecOrMat) =
    b .*= chol.M.diag

@inline LinearAlgebra.:\(chol::BlockCholesky, b::AbstractVecOrMat) = chol.F\b

@inline LinearAlgebra.:\(chol::BlockCholesky{<:Any,<:Diagonal}, b::AbstractVecOrMat) =
    chol.M*b




# Cost function interface
struct InvertedQuadratic{n,m,T,S}
    chol::BlockCholesky{T,S}
    q::MVector{n,T}
    r::MVector{m,T}
end

function InvertedQuadratic(cost::DiagonalCost)
    n = state_dim(cost)
    m = control_dim(cost)
    m̄ = m * !cost.terminal
    chol = BlockCholesky(n, m̄, diag=true)
    icost = InvertedQuadratic(chol, MVector(cost.q), MVector(cost.r))
    update_cost!(icost, cost)
    icost
end

function InvertedQuadratic(cost::QuadraticCost{<:Any,<:Any,<:Any,TQ,TR}) where {TQ,TR}
    n = state_dim(cost)
    m = control_dim(cost)
    m̄ = m * !cost.terminal
    if cost.zeroH && TQ <: Diagonal && TR <: Diagonal
        chol = BlockCholesky(n, m̄, diag=true)
    else
        M = zeros(n+m̄,n+m̄)
        chol = BlockCholesky(M, n,m̄, block_diag=cost.zeroH)
    end
    icost = InvertedQuadratic(chol, cost.q, cost.r)
    update_cost!(icost, cost)
    icost
end

function gradient(icost::InvertedQuadratic{<:Any,m}) where m
    if size(icost.chol.B,1) == m
        return [SVector(icost.q); SVector(icost.r)]
    else  # terminal
        return SVector(icost.q)
    end
end

function add_gradient!(z, icost::InvertedQuadratic{n,m}) where {n,m}
    for i = 1:n
        z[i] += icost.q[i]
    end
    if length(z) == n+m
        for i = 1:m
            z[n+i] += icost.r[i]
        end
    end
end

function update_cost!(icost::InvertedQuadratic, cost::QuadraticCost)
    if cost.zeroH
        cholesky!(icost.chol, cost.Q, cost.R)
    else
        cholesky!(icost.chol, cost.Q, cost.R, cost.H')
    end
    icost.q .= cost.q
    icost.r .= cost.r
end

function update_cost!(icost::InvertedQuadratic, cost::DiagonalCost)
    cholesky!(icost.chol, cost.Q, cost.R)
    icost.q .= cost.q
    icost.r .= cost.r
end

function update_cholesky!(chols::Vector{<:InvertedQuadratic}, obj::Objective)
    for k in eachindex(chols)
        update_cost!(chols[k], obj.cost[k])
    end
end
