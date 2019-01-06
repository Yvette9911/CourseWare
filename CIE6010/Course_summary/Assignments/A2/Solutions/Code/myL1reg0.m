function x = myL1reg0(A, b, D)
% This function solves the optimizatin problem
% Input:
%       A: a m*n matrix
%       b: a m*1 vector
%       D: a k*n matrix
% minimize sum_{i=1}^m t_i
% such that A * x = b
% for i = 1:k,
%        - t_i \le \sum_{j=1}^n d_{ij}x_j \le t_i
% construct decision variable X = [x_1,...,x_n,s_1,...,s_m]';
%% Estimate size
[m,n] = size(A);
[k,~] = size(D);

%% Construct f, Aineq, bineq, Aeq, beq
f(n+1:n+k) = 1;
Aineq = [D,-1 * eye(k); -1 *D, -1 * eye(k)];
bineq = zeros(k+k,1);
Aeq = [A,zeros(m,k)];
beq = b;
%% Use Linprog to solve optimization
options = optimoptions('linprog','Algorithm','interior-point','MaxIter',20,...
    'OptimalityTolerance',5e-4,'ConstraintTolerance',1e-4,'Display','off');
X = linprog(f,Aineq,bineq,Aeq,beq,[],[],options);

x = X(1:n);

end 