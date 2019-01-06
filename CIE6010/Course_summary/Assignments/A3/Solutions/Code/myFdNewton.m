function [ x,iter ] = myFdNewton( func,x0,tol,maxit,varargin )
%  Implement Newton's method using finite difference approximation
%  Input:
%   func: function handle
%     x0: initial guess
%    tol: tolerance
%  maxit: maximum iteration
%   varargin - parameters
% Output:
%      x: optimal solution
%   iter: total number of iterations
%% First Iteration
x = x0;
g = feval(func,x,varargin{:});nrmg0 = norm(g);
fprintf('iter: %3d  nrmg/nrmg0 = %6.2e\n',1,1)
H = feval('myHessian',func,x,g,varargin{:});
x = x - H \ g;
%% Remaining iterations
for iter = 2:maxit
    g = feval(func, x, varargin{:});   % get gradient g and Hessian H
    fraction = norm(g)/nrmg0;
    fprintf('iter: %3d  nrmg/nrmg0 = %6.2e\n',iter,fraction)
    if fraction <= tol,break;end
    H = feval('myHessian',func,x,g,varargin{:});
    x = x - H \ g;
end


end


function H = myHessian(func,x,gx,varargin)
%delta = 1e-9;
m = length(gx); n = length(x);
H = zeros(m,n);
for j = 1:n
    epsilon = 1e-9 *  max(1,max(1,abs(x(j))) * sign(x(j)));
    x(j) = x(j) + epsilon;
    g = feval(func, x, varargin{:});
    x(j) = x(j) - epsilon;
    H(:,j) = (g - gx) / epsilon;
end

end