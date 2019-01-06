function [ g,H,f ] = myQuartic( x,A,u,mu )
% Output:
%      g: gradient, g = (x'*A*x - 1/4*u'*A*u) * A*x + mu * (x - u)
%      H: Hessian, H = (x'*A*x - 1/4*u'*A*u) * A + 2*A*x*x'*A + mu * eye(n)
%      f: function itself, 
%      f = 1/4*(x'*A*x - 1/4*u'*A*u) + mu/2 * ||x - u||_2^2

n = length(x);

Ax = A*x;
quartic = x'*Ax - u'*A*u/4;
y = x - u;

%% Evaluate g
g = quartic * Ax + mu * y;
if nargout < 2;return;end

%% Evaluate H
H = quartic * A + 2 * (Ax * Ax') + mu * speye(n);
if nargout < 3;return;end

%% Evaluate f
f = quartic/4 + mu/2 * (y'*y);

end