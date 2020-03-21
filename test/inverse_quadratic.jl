using StaticArrays
using LinearAlgebra
using TrajOptCore
using BenchmarkTools
using Test

n,m = 10,5
x = @SVector rand(n)
u = @SVector rand(m)
z = KnotPoint(x,u,0.1)

A = zeros(n,n)
A = Diagonal(@MVector ones(n))
dinds = diag(LinearIndices(a))[1:3]
inds = 1:3
A.diag[inds]
@btime $A.diag[inds] .= $b
A

# Test hessian inverse
Q = Diagonal(@SVector rand(n))
R = Diagonal(@SVector rand(m))
costfun = QuadraticCost(Q,R)
iv = InvertedQuadratic(n,m,true)
LinearAlgebra.inv!(iv, costfun)
LinearAlgebra.inv!

A = rand(10,10)
A = A'A
b = rand(10)

# Get cholesky factor directly
Achol = copy(A)
Chol = Cholesky(Achol,'U',0)
LAPACK.potrf!('U',Achol)
b1 = copy(b)
ldiv!(Chol,b1)
b2 = copy(b)
LAPACK.potrs!('U',Achol,b2)
bs = SVector{10}(b)
b1 ≈ b2
b1 ≈ Chol\b
b1 ≈ A\b
Chol\bs

@btime ldiv!($Chol,$b)
@btime LAPACK.potrs!('U',$Achol,$b2)
@btime $Chol\$b
@btime $Chol\$bs
@btime $A\$b
typeof('U')

@btime zeros(0,0)

typeof(Chol)
A = Diagonal(@MVector rand(10))
b = rand(10)
b = @SVector rand(10)

Achol = copy(A)
Chol = Cholesky(Achol, 'U', 0)
Chol\b

Cholesky
a = view(A,1:3,1:3)


LinearAlgebra.inv!(A,B)
J = Diagonal([diag(Q); diag(R)])
z_ = costfun\z
@test J\z.z ≈ z_.z

costfun = DiagonalCost(Q,R)
J = Diagonal([diag(Q); diag(R)])
z_ = costfun\z
@test J\z.z ≈ z_.z

Q = rand(n,n)
R = rand(m,m)
Q = SizedMatrix{n,n}(Q'Q)
R = SizedMatrix{m,m}(R'R)
costfun = QuadraticCost(Q,R)
@test costfun.zeroH
@test_throws UndefRefError costfun.Sinv
J = cat(Q,R,dims=[1,2])
z_ = costfun\z
@test J\z.z ≈ z_.z

H = SizedMatrix{n,m}(rand(n,m))
costfun = QuadraticCost(Q,R,H=H)
@test !costfun.zeroH
J = [Q H; H' R]
z_ = costfun\z
@test z_.z ≈ J\z.z
@btime $costfun\$z
