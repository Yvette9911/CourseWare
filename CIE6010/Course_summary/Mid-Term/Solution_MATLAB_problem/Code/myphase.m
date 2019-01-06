function [ x,hist,iter ] = myphase(A,B,d,x0,tol,maxit,itGN )
% Usage:
% Solve the phase retrieval least squares problem:
%       min f(x) = 0.5 * norm(|(A + iB)x|^2 - d)^2
%
% Input:
%      A, B, d = real-valued data
%           x0 = initial guess
%          tol = tolerance
%        maxit = maximum number of iterations in total
%         itGN = min number of Gauss-Newton iterations before
% Output:
%            x = computed solution
%         hist = history vector for c_k = norm(g_k) / (1 + f_k)
%         iter = total number of iterations taken
x = x0;
hist = [];
itGN = itGN*5;
for iter = 1:maxit
    [J,r] = Jacobian_compute(A,B,d,x);
    nabla = J' * r;
    nabla2_fake = J' * J;
    f = norm(r)^2*0.5;
    gnrm = norm(nabla);
    crit = gnrm / (1 + f);
    fprintf('iter %i: f(x) = %9.2e gnrm = %9.2e crit = %9.2e\n',...
       iter, f, gnrm, crit)
    hist(iter) = crit;
    if crit <= tol, break;end
    if iter <= itGN
        m = nabla2_fake \ nabla;
        x = x - m;
    else
        S = 2*(r'.*A'*A+r'.*B'*B);
        nabla2 = nabla2_fake + S;
        m = nabla2 \ nabla;
        x = x - m;
    end
end
end

function [J,r] = Jacobian_compute(A,B,d,x)
Ax = A*x;
Bx = B*x;
r = Ax.^2 + Bx.^2-d;
J = 2*A.*Ax+2*B.*Bx;
end
