
struct DPSolver{n,m,T} <: AbstractSolver
    A::SizedMatrix{n,n,T,2}
    B::SizedMatrix{n,m,T,2}
    P::SizedMatrix{n,n,T,2}
    P_::SizedMatrix{n,n,T,2}
    PA::SizedMatrix{n,n,T,2}
    PB::SizedMatrix{n,m,T,2}
    APB::SizedMatrix{n,m,T,2}
    E::SizedMatrix{m,m,T,2}
end

function DPSolver(prob::LQRProblem{n,m}) where {n,m}
    A = copy(prob.A)
    B = copy(prob.B)
    P = zero(A)
    P_ = zero(A)
    PA = zero(A)
    PB = zero(B)
    APB = zero(B)
    E = SizedMatrix{m,m}(zeros(m,m))
    DPSolver(A,B, P,P_, PA,PB,APB,E)
end

@inline _get_data(A::AbstractArray) = A
@inline _get_data(A::SizedArray) = A.data

function chol_solve!(A,b)
    LAPACK.potrf!('U', _get_data(A))
    LAPACK.potrs!('U', _get_data(A), _get_data(b))
end


compute_gain!(K, solver::DPSolver, prob::LQRProblem) =
    compute_gain!(K, solver.P, prob.A, prob.B, prob.Q, prob.R,
        solver.PA, solver.PB, solver.APB, solver.E)
function compute_gain!(K, P, A, B, Q, R, PA, PB, APB, E)
    PB .= P*B
    E .= R .+ B'PB
    PA .= P*A
    K .= B'PA
    chol_solve!(E,K)
end

@inline compute_ctg!(K, solver::DPSolver, prob::LQRProblem) =
    compute_ctg!(solver.P_, K, solver.P, prob.A, prob.B, prob.Q, prob.R,
        solver.PA, solver.PB, solver.APB, solver.E)
function compute_ctg!(P_, K, P, A, B, Q, R, PA, PB, APB, E)
    compute_gain!(K, P, A, B, Q, R, PA, PB, APB, E)
    APB .= A'PB
    P_ .= Q .+ A'PA .- APB*K
end

function solve!(sol::LQRSolution, solver::DPSolver, prob::LQRProblem)
    N = prob.N

    # Terminal ctg
    solver.P .= prob.Qf

    # Riccati BP
    for k = N-1:-1:1
        compute_ctg!(sol.K[k], solver, prob)
        solver.P .= solver.P_
    end

    sol.X[1] .= prob.x0
    for k = 1:N-1
        sol.U[k] .= -sol.K[k] * sol.X[k]
        sol.X[k+1] .= prob.A*sol.X[k] + prob.B*sol.U[k]
    end

end
