function LinearAlgebra.cholesky!(L::Vector{<:BlockTriangular3}, F::Vector{<:BlockTriangular3})
    N = length(F)
    A = cholesky(F[1].A)
    for k = 1:N
        if k < N
            F[k].C .+= F[k+1].A
            F[k+1].A .*= 0
        end
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

function backward_substitution!(L::Vector{<:BlockTriangular3})
    N = length(L)
    # Handle terminal case
    backward_substitution!(L[N], L[N])
    for k = N-1:-1:1
        backward_substitution!(L[k], L[k+1])
    end
end

function backward_substitution!(L::BlockTriangular3, Lprev::BlockTriangular3)
    backward_substitution!(L, Lprev.λ)
end

function backward_substitution!(L::BlockTriangular3, λprev)
    μ,λ = L.μ, L.λ
    c,d = L.c, L.d

    μ .= L.B'\(c - L.E'λprev)
    λ .= L.C'\(d - L.D'μ - L.F'λprev)
end
