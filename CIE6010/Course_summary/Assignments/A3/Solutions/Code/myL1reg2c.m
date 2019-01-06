function [ x,iter ] = myL1reg2c( A,b,D )
% Input: 
%     A: a m*n matrix
%     b: a m*1 vector
%     D: a k*n matrix
% Usage:
%     solve the unconstrained minimization model
%     min phi_sigma(D * x) + mu/2 * ||A*x - b||_2^2
% with
% sigma = 0.05 around
% mu = 0.1 around
% phi_sigma(y) = \sum_{i=1}^k(y_i^2+sigma)^(1/2)
%% parameters setting
[~,n] = size(D);
global sigma mu alpha
tol_2 = 1e-8;
maxiter = 50000;    
%% initial setting 
x = sparse(n,1);  
x_p = sparse(n,1);
Dx = D*x;  
[f,g] =  my_ob_nabla(x,Dx,D,A,b,sigma,mu);  
gnorm = norm(g);
tol_1 = gnorm * 1e-2;
a_p = 0;  a = 1;
%% Iteration Running
for iter = 1:maxiter
    m = 0.5 * (1+sqrt(1+4*a^2)); a_p = a; a = m;
    t = (a_p - 1) / a;
    y = (1+t) * x - t * x_p;    Dy = D*y;
    % update function
    x_p = x;    [f_try,nabla] = my_ob_nabla(y,Dy,D,A,b,sigma,mu);
    x = y - alpha * nabla;
    f_diff = abs(1 - f_try / f);    f = f_try;  
    gnorm = norm(nabla);
    if f_diff <= tol_2 &&  gnorm <= tol_1, break; end
end
end

%% Computing object and nabla
function [object,nabla] = my_ob_nabla(x,Dx,D,A,b,sigma,mu)

K = sqrt(((Dx).^2+sigma));
Y = A*x - b;
object = sum(K) + mu/2 * norm(Y)^2;
if nargin == 1, return;end
nabla = ((Dx./K)' * D)'+(mu*Y' * A)';

end