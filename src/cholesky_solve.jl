function LinearAlgebra.cholesky!(L::Vector{<:BlockTriangular3}, F::Vector{<:BlockTriangular3})
    N = length(F)
    A = cholesky(F[1].A)
    for k = 1:N
        A = cholesky!(L[k], F[k], A)
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
    return C
end

function forward_substitution!(L::Vector{<:BlockTriangular3})
    N = length(L)
    forward_substitution!(L[1])
    for k = 2:N
        forward_substitution!(L[k-1],L[k])
    end
end

function forward_substitution!(L1::BlockTriangular3, L2::BlockTriangular3)
    L2.μ .= L2.B\(L2.c - L2.D*L1.λ)
    L2.λ .= L2.C\(L2.d - L2.F*L1.λ - L2.E*L2.μ)
end

function forward_substitution!(L::BlockTriangular3)
    L.μ .= L.B\L.c
    L.λ .= L.C\(L.d - L.E*L.μ)
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
    L.λ .= L.C'\(L.λ - Lprev.D'*Lprev.μ - Lprev.F'Lprev.λ)
    L.μ .= L.B'\(L.μ - L.E'L.λ)
end

function backward_substitution!(L::BlockTriangular3)
    L.μ .= L.B\L.μ
end
