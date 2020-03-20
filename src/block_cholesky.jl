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
        BlockCholesky(Diagonal(@MMatrix zero(n+m), n,m; kwargs...))
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

    A_ = Cholesky(zeros(n,n),uplo,0)
    B_ = Cholesky(zeros(m,m),uplo,0)
    F = Cholesky(M, uplo, 0)
    BlockCholesky(M, F, A, B, C, Ct, A_, B_, block_diag, uplo)
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
    n = size(A,1)
    m = size(B,1)
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
