function [X, iter] = myGN(A,X0,tol,maxiter)
% Input:
%        A: given matrix
%       X0: initial guess
%      tol: tolerance
%  maxiter: maximum iterations
%Output:
%        X: solution to the opt
%     iter: number of iterations

k = size(X0,2);
I = speye(k);
for iter = 1:maxiter
    M = X0'*X0;
    Y = X0/M;
    Z = A*Y;
    X = Z - X0 * ((Y'*Z-I)/2);
    if norm(X - X0,'fro')^2 <= tol^2 * trace(M),break;end
    X0 = X;
end
end