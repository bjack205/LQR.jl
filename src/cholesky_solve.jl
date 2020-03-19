function chol!(A::SizedArray, uplo='U')
    LAPACK.potrf!(uplo, A.data)[2]
end

function LinearAlgebra.cholesky!(L::Vector{<:BlockTriangular3}, F::Vector{<:BlockTriangular3})
    N = length(F)
    A = cholesky(F[1].A)
    # A = chol!(F[1].A)
    for k = 1:N
        A = cholesky!(L[k], F[k], A)
        # A = cholesky2!(L[k], F[k], A)
    end
end

function LinearAlgebra.cholesky!(L::BlockTriangular3, F::BlockTriangular3, A)
    L.D .= F.D/(A.U)
    B = cholesky(F.B - L.D*L.D')
    L.B .= B.L
    L.F .= F.F/(A.U)
    L.E .= (F.E - L.F*L.D')/(B.U)
    C = cholesky(F.C - L.F*L.F' - L.E*L.E')
    L.C .= C.L
    L.c .= F.c
    L.d .= F.d
    return C
end

function LinearAlgebra.cholesky!(U::Vector{<:BlockUpperTriangular3}, F::Vector{<:BlockUpperTriangular3})
    N = length(F)
    for k = 1:N
        cholesky!(U[k], F[k])
    end
end

@inline get_data(A::AbstractArray) = A
@inline get_data(A::SizedArray) = A.data
function tri_solve!(A,b::AbstractVector, uplo='U', trans='N', diag='N')
    # LAPACK.trtrs!(uplo, trans, diag, get_data(A), get_data(B))
    BLAS.trsv!(uplo, trans, diag, get_data(A), get_data(b))
end

function tri_solve!(A,B::AbstractMatrix, uplo='U', trans='N', diag='N')
    # LAPACK.trtrs!(uplo, trans, diag, get_data(A), get_data(B))
    BLAS.trsm!('L', uplo, trans, diag, 1.0, get_data(A), get_data(B))
end

function LinearAlgebra.cholesky!(U::BlockUpperTriangular3, F::BlockUpperTriangular3)
    U.D .= F.D
    tri_solve!(U.A, U.D, 'U', 'T')
    U.B .= F.B .- U.D'U.D

    U.B .= F.B .- U.D'U.D
    st = chol!(U.B, 'U')
    # U.B .= UpperTriangular(U.B)

    U.F .= F.F
    tri_solve!(U.A, U.F, 'U', 'T')

    U.E .= F.E .- U.D'U.F
    tri_solve!(U.B, U.E, 'U', 'T')
    U.C .= F.C .- U.F'U.F .- U.E'U.E
    chol!(U.C, 'U')
    # U.C .= UpperTriangular(U.C)
    U.c .= F.c
    U.d .= F.d
    return nothing
end

function LinearAlgebra.cholesky!(F::Vector{<:BlockUpperTriangular3})
    N = length(F)
    for k = 1:N
        cholesky!(F[k])
    end
end

function LinearAlgebra.cholesky!(U::BlockUpperTriangular3)
    tri_solve!(U.A, U.D, 'U', 'T')

    U.B .-= U.D'U.D
    st = chol!(U.B, 'U')
    # U.B .= UpperTriangular(U.B)

    tri_solve!(U.A, U.F, 'U', 'T')

    U.E .-= U.D'U.F
    tri_solve!(U.B, U.E, 'U', 'T')
    U.C .-= (U.F'U.F .+ U.E'U.E)
    chol!(U.C, 'U')
    # U.C .= UpperTriangular(U.C)
    return nothing
end

function forward_substitution!(L1::BlockUpperTriangular3, L2::BlockUpperTriangular3)
    L2.μ .= L2.c .- L2.D'L1.λ
    tri_solve!(L2.B, L2.μ, 'U', 'T')
    # L2.μ .= L2.B'\L2.μ
    L2.λ .= L2.d .- L2.F'L1.λ .- L2.E'L2.μ
    # L2.λ .= L2.C'\L2.λ
    tri_solve!(L2.C, L2.λ, 'U', 'T')
end

function forward_substitution!(L::BlockUpperTriangular3)
    L.μ .= L.c
    tri_solve!(L.B, L.μ, 'U', 'T')
    # L.μ .= L.B'\L.c
    L.λ .= L.d .- L.E'L.μ
    tri_solve!(L.C, L.λ, 'U', 'T')
    # L.λ .= L.C'\L.λ
end

function backward_substitution!(L::BlockUpperTriangular3, Lprev::BlockUpperTriangular3)
    L.λ .= L.λ .- Lprev.D*Lprev.μ .- Lprev.F*Lprev.λ
    # L.λ .= L.C\L.λ
    tri_solve!(L.C, L.λ, 'U')
    L.μ .= L.μ .- L.E*L.λ
    # L.μ .= L.B\L.μ
    tri_solve!(L.B, L.μ, 'U')
end

function backward_substitution!(L::BlockUpperTriangular3)
    # L.μ .= L.B\L.μ
    tri_solve!(L.B, L.μ, 'U')
end


function forward_substitution!(L::Vector{<:BlockTriangular3})
    N = length(L)
    forward_substitution!(L[1])
    for k = 2:N
        forward_substitution!(L[k-1],L[k])
    end
end

function forward_substitution!(L1::BlockTriangular3, L2::BlockTriangular3)
    L2.μ .= L2.c .- L2.D*L1.λ
    L2.μ .= L2.B\L2.μ
    L2.λ .= L2.d .- L2.F*L1.λ .- L2.E*L2.μ
    L2.λ .= L2.C\L2.λ
end

function forward_substitution!(L::BlockTriangular3)
    L.μ .= L.B\L.c
    L.λ .= L.d .- L.E*L.μ
    L.λ .= L.C\L.λ
end

function backward_substitution!(L::Vector{<:BlockTriangular3})
    N = length(L)
    # Handle terminal case
    backward_substitution!(L[N])
    for k = N-1:-1:1
        backward_substitution!(L[k], L[k+1])
    end
end

function backward_substitution!(L::BlockTriangular3, Lprev::BlockTriangular3)
    L.λ .= L.λ .- Lprev.D'*Lprev.μ .- Lprev.F'Lprev.λ
    L.λ .= L.C'\L.λ
    L.μ .= L.μ .- L.E'L.λ
    L.μ .= L.B'\L.μ
end

function backward_substitution!(L::BlockTriangular3)
    L.μ .= L.B'\L.μ
end
