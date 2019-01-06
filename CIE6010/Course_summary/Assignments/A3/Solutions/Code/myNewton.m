function [x, iter ] = myNewton(func, x0, tol, maxit, varargin )
% Implement Newton's method using analytic Jacobian
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
[g,H] = feval(func, x, varargin{:});
nrmg0 = norm(g);
x = x - H \ g;
fprintf('iter: %3d  nrmg/nrmg0 = %6.2e\n',1,1)
%% Remaining iterations
for iter = 2:maxit
    [g, H] = feval(func, x, varargin{:});   % get gradient g and Hessian H
    fraction = norm(g)/nrmg0;
    fprintf('iter: %3d  nrmg/nrmg0 = %6.2e\n',iter,fraction)
    if fraction <= tol,break;end
    x = x - H \ g;
end

end