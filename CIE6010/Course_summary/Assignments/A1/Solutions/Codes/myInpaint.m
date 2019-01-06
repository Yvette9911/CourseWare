function [ X ] = myInpaint( Xh, Omega )
%      Usage:
%      Input:
%         Xh: the available matrix, with size n*n
%      Omega: the available pixel set Omega, with size m
%     Output:
%          X: the recovered graph

%% Figure Out relevant sizes
n = length(Xh);
m = length(Omega);

%% Construct the E matrix and the S matrix
I = speye(n,n);
% Generate vectors of subscripts and corresponding values of D
i = 1:n-1;
%j = 2:n;
v = -1 * ones(n-1,1);
D = sparse(i,i+1,v,n-1,n) + speye(n-1,n);
E = [kron(I,D);kron(D,I)];
% Generate vectors of subscripts and corresponding values of S
i = 1:m;
j = Omega;
v = 1 * ones(m,1);
S = sparse(i,j,v,m,n^2);

%% Construct the A matrix, f and b vectors
[v,~] = size(E);
[w,~] = size(S);
I = speye(v);
A = [E,-I,I;S,sparse(w,v),sparse(w,v)];
vecXh = Xh(:);
xh_Omega = vecXh(Omega);
b = [sparse(v,1);xh_Omega];
f = [sparse(n^2,1);ones(v * 2,1)]';

%% Solving the linear programming
LB = sparse(n^2 + 2*v,1);
UB = Inf * ones(n^2 + 2*v,1);
options = optimoptions('linprog','Algorithm','interior-point');
z = linprog(f,[],[],A,b,LB,UB,options);
%% Extract vector x and reshape into matrix X
x = z(1:n^2);
X = reshape(x,[n,n]);
end