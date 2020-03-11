
model = RobotZoo.DubinsCar()
n,m = size(model)
N,dt = 101, 0.01
NN = N*n + (N-1)*m
Z = zeros(NN)
ix = @SVector [1,2,3]
iu = @SVector [4,5]

z = ViewKnotPoint(view(Z,1:5), ix, iu, dt, 0.0)
state(z)
Z[1] = 10
state(z)[1] == 10
