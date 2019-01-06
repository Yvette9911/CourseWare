function [x,iter] = myL1reg1(A, b, D)
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
sigma = 5e-2;   mu = 1e-1;
object = @(x,Dx) sum(sqrt(((Dx).^2+sigma))) + mu/2 * norm(A*x - b)^2;
nabla = @(x,Dx) ((Dx./sqrt(Dx.^2 + sigma))'*D)'+(mu*(A*x - b)' * A)';
tol_2 = 1e-7;
maxiter = 50000;    
beta = 0.5;   
C1 = 1e-5;
%% initial setting
x = A' * ((A*A')\b); 
Dx = D*x;
f = object(x,Dx);   
g = nabla(x,Dx);   
gnorm = norm(g);    %gnorm0 = gnorm
tol_1 = gnorm * 5e-2;
alpha = 1;  %initial step-size

%% Iteration Running
for iter = 1:maxiter
    delta = C1 * alpha * gnorm^2;
    Dg = D*g;
    for arm = 1:10
        % Armijo Condition
        x_try = x - alpha * g;
        Dx_try = Dx - alpha * Dg;
        f_try = object(x_try,Dx_try);
        if f_try <= f - delta,break; end
        alpha = alpha * beta;
    end
    % update function
    x_pre = x;  g_pre = g;    f_diff = 1 - f_try/f;
    x = x_try;  Dx = Dx_try;  g = nabla(x,Dx); 
    f = f_try;  gnorm = norm(g);
    if gnorm <= tol_1   && f_diff <= tol_2, break; end
    % BB step
    s = x - x_pre;  y = g - g_pre;
    alpha = (s'*y) / (y' * y);
end
end