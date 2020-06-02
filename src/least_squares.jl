
struct LeastSquaresSolver{n,m,T,M,V} <: AbstractSolver
    H::M
    y::V
    Ā::M
    b̄::V
    T::M  # dynamics matrix
    L::M  # IC matrix
    Hx::M # state cost matrix
    Hu::M # control cost matrix
    tmpA::SizedMatrix{n,n,T,2}
    tmpB::SizedMatrix{n,m,T,2}
    tmpA2::SizedMatrix{n,n,T,2}
    tmpB2::SizedMatrix{n,m,T,2}
    Qf::SizedMatrix{n,n,T,2}  # cholesky factor of Qf
    Q::SizedMatrix{n,n,T,2}   # cholesky factor of Q
    R::SizedMatrix{m,m,T,2}   # cholesky factor of R
    opts::Dict{Symbol,Symbol}
end

function _get_zeros_fun(ArrayType)
    if ArrayType <: Array
        _zeros = zeros
    elseif ArrayType <: SparseMatrixCSC
        _zeros = spzeros
    end
    return _zeros
end

function LeastSquaresSolver(prob::LQRProblem, MatType=Matrix, VecType=Vector)
    n,m,N = size(prob)
    Nn,Nm = N*n, (N-1)*m

    mat_zeros = _get_zeros_fun(MatType)
    vec_zeros = _get_zeros_fun(VecType)

    H = mat_zeros(Nm,Nm)
    y = vec_zeros(Nm)
    A = mat_zeros(Nn, Nm)
    b = vec_zeros(Nn)
    T = mat_zeros(Nn, Nm)
    L = mat_zeros(Nn, n)
    Hx = mat_zeros(Nn, Nn)
    Hu = mat_zeros(Nm, Nm)

    tmpA = SizedMatrix{n,n}(zeros(n,n))
    tmpB = SizedMatrix{n,m}(zeros(n,m))
    tmpA2 = SizedMatrix{n,n}(zeros(n,n))
    tmpB2 = SizedMatrix{n,m}(zeros(n,m))
    Qf = SizedMatrix{n,n}(cholesky(prob.Qf).U)
    Q  = SizedMatrix{n,n}(cholesky(prob.Q).U)
    R  = SizedMatrix{m,m}(cholesky(prob.R).U)
    opts = Dict(:solve_type=>:cholesky,
                :matbuild=>:Ab)
    LeastSquaresSolver(H,y,A,b,T,L,Hx,Hu,tmpA,tmpB,tmpA2,tmpB2,Qf,Q,R,opts)
end

function buildAb!(solver::LeastSquaresSolver, prob::LQRProblem{n,m}) where {n,m}
    Ā,b̄ = solver.Ā, solver.b̄
    A,B = prob.A, prob.B
    tmpA, tmpA2 = solver.tmpA, solver.tmpA2
    tmpB, tmpB2 = solver.tmpB, solver.tmpB2
    T,L = solver.T, solver.L
    ix,iu = 1:n, 1:m
    N = prob.N-1
    x0 = prob.x0

    tmpB .= B
    Bi = tmpB2
    An = tmpA2
    An .= Diagonal(@SVector ones(n))

    for j = 0:N          # loop powers of A
        if j < N
            Qi = UpperTriangular(solver.Q)
        else
            Qi = UpperTriangular(solver.Qf)
        end
        ix_ = ix .+ j*n
        iu_ = iu .+ j*m
        tmpA .= Qi * An
        bj = tmpA * x0
        b̄[ix_] .= bj

        tmpB .= An * B
        for i = 1:(N-j)       # loop over columns
            row = i + j       # 0-based index
            ix_ = ix .+ (i+0)*n .+ j*n
            iu_ = iu .+ (i-1)*m
            # @show ix_, iu_
            # mul!(tmpB, tmpA, B)
            if row < N
                Qi = solver.Q
            else
                Qi = solver.Qf
            end
            Bi .= Qi * tmpB
            Ā[ix_, iu_] .= Bi
        end
        # mul!(An, A, An)
        An .= A*An
    end
end

function build_least_squares!(solver::LeastSquaresSolver, prob::LQRProblem{n,m}) where {n,m}
    A,B = prob.A, prob.B
    tmpA = solver.tmpA
    tmpB = solver.tmpB
    T,L = solver.T, solver.L
    ix,iu = 1:n, 1:m
    N = prob.N-1

    tmpA .= Diagonal(@SVector ones(n))
    tmpB .= B
    for j = 0:N          # loop powers of A
        ix_ = ix .+ j*n
        iu_ = iu .+ j*m
        L[ix_,ix] .= tmpA
        if j < N
            solver.Hx[ix_, ix_] = solver.Q
            solver.Hu[iu_, iu_] = solver.R
        else
            solver.Hx[ix_, ix_] = solver.Qf
        end

        for i = 1:(N-j)       # loop over columns
            ix_ = ix .+ (i+0)*n .+ j*n
            iu_ = iu .+ (i-1)*m
            mul!(tmpB, tmpA, B)
            T[ix_, iu_] .= tmpB
        end
        mul!(tmpA, A, tmpA)
    end
end

function build_toeplitz(T, L, A, B, tmpA=copy(A), tmpB=copy(B))
    n,m = size(B)
    ix,iu = 1:n, 1:m
    N = Int(size(L,1)/n)-1

    tmpA .= Diagonal(@SVector ones(n))
    tmpB .= B
    for j = 0:N          # loop powers of A
        ix_ = ix .+ j*n
        iu_ = iu .+ j*m
        L[ix_,ix] .= tmpA

        for i = 1:(N-j)       # loop over columns
            ix_ = ix .+ (i+0)*n .+ j*n
            iu_ = iu .+ (i-1)*m
            mul!(tmpB, tmpA, B)
            T[ix_, iu_] .= tmpB
        end
        mul!(tmpA, A, tmpA)
    end
end

function solve!(sol::Primals, solver::LeastSquaresSolver, prob::LQRProblem)
    H,y = solver.H, solver.y
    A,b = solver.Ā, solver.b̄
    if solver.opts[:matbuild] == :Ab
        buildAb!(solver, prob)
    elseif solver.opts[:matbuild] == :lsq
        build_least_squares!(solver, prob)
        mul!(A, solver.Hx, solver.T)
        b0 = copy(b)
        mul!(b0, solver.L, prob.x0)
        mul!(b, solver.Hx, b0)
    end

    mul!(H, A', A)
    H .+= solver.Hu
    mul!(y, A', b, -1.0, 0.0)
    # F = qr!(H)
    # F = cholesky!(H)
    # ldiv!(F,-y)
    if solver.opts[:solve_type] == :cholesky
        if typeof(A) <: SparseMatrixCSC
            sol.U_ .= cholesky(H)\y
        else
            LAPACK.potrf!('U', H)
            LAPACK.potrs!('U', H, y)
            sol.U_ .= y
        end
    elseif solver.opts[:solve_type] == :naive
        sol.U .= -H\y
    end
    rollout!(sol, prob)
    return nothing
end

@inline traj(Z::Primals) = Z.Z_
@inline vect(Z::Primals) = Z.Z

@inline TO.rollout!(sol::Primals, prob::LQRProblem) =
    rollout!(sol, prob.A, prob.B, prob.x0)
function TO.rollout!(sol::Primals, A, B, x0)
    sol.X[1] .= x0
    for k in eachindex(sol.U)
        sol.X[k+1] .= A * sol.X[k] + B * sol.U[k]
    end
end
