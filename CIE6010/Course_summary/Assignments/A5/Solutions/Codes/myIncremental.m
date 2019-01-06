function x = myIncremental(A,b,x0,tol,maxit)
%   Input:
%       A: given coefficient matrix
%       b: given cost vector
%      x0: initial guess
%     tol: tolerance
%   maxit: maximum iteration
%  Output:
%       x: least squares solution
[m,~] = size(A);
x = x0;
theta = 40;
g = @(ai,x,bi)(ai*(ai'*x-bi));
A = A';
for iter = 1:maxit
    alpha = theta/iter;
    psi = x;
     for j = 1:m
         ai = A(:, j);
         bi = b(j);
         psi = psi - alpha * g(ai,psi,bi);
     end
    %psi = psi - alpha * A' * (A*psi - b);
    Delta = norm(x-psi)/ norm(x);
    if Delta <= tol 
        fprintf('iter %2i: alpha = %9.2e  Delta = %9.2e\n',...
            iter,alpha, Delta)
        break;
    end
    if mod(iter,50) == 0
        fprintf('iter %2i: alpha = %9.2e  Delta = %9.2e\n',...
            iter,alpha, Delta);
    end
    x = psi;
end
end

