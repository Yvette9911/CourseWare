function [ x,y,lamb,out ] = my_qcqp_admm( A,b,c,tol )
%% usage
% solving the QCQP problem
% ------------------------
% min x'*A*x/2 - b'*x + y'*A*y/2 - c'*y
% s.t. (x'*x-1)/2=0,  (y'*y-1)/2=0, x'*y=0
% ------------------------
%
%% variables
% input:
%        (A,b,c) = QCQP problem data
%            tol = tolerance for stationarity
%
% output:
%          (x,y) = computed solutions
%          lamb  = [1 by 3] 3 multipliers
%           out  = a struct with fields:
%           iter : number of iterations taken
%      converged : true or false
%           rres : [iter by 3] 3 relative residual
%                  quantities at every iterations

%% Parameters Setting
maxiter = 1000;
n = length(A);
Asiz = norm(A,'fro')/sqrt(n);
s = norm(b)/Asiz;
fx = @(x)A*x;
if s > 5
    tau =(sqrt(5)+1)/2;
    rho = s*15.5;
    y=c;y = y/norm(y);  % initial guess
    tol_eig = 1e-6;
elseif s > 0.2
    tau = 1.95;%(sqrt(5)+1)/2;
    rho = s*5.375;
    y=c;y = y/norm(y);  % initial guess
    tol_eig = 1e-6;
else
    tau = 1.95;%(sqrt(5)+1)/2;
    rho = s*.0875;
    if n <= 10000,rho = rho * 10;end
    [y,~] = eigs(@(x)fx(x),n,1,'lm');   % initial guess
    tol_eig = 1e-10;
end
lamb = zeros(3,1);  % initialize multipliers
lamb(3) = y'*b;
v1 = rand(2*n,1);v1 = v1/norm(v1);
v2 = v1;
%% iterations
for iter = 1:maxiter
    if iter >= 3,tol_eig = 1e-12;end
    [x,lamb(1),v1] = my_TRS_solver(A,b,lamb(3),y,rho,tol_eig,v1);
    lamb(3) = lamb(3) +  tau* rho * x'*y;
    [y,lamb(2),v2] = my_TRS_solver(A,c,lamb(3),x,rho,tol_eig,v2);
    inn_xy = x' * y;
    lamb(3) = lamb(3) +  tau* rho * inn_xy;
    norm12 = [norm(fx(x) - b + lamb(3) * y + lamb(1) * x + rho * y * inn_xy),...
              norm(fx(y) - c + lamb(3) * x + lamb(2) * y + rho * x* inn_xy)];
    norm12 = norm12./([1+norm(b),1+norm(c)]);
    norm3 = norm([0.5*(x'*x-1); 0.5*(y'*y-1); x'*y]);
    out.rres(iter,:) = [norm12,norm3];
    fprintf('iter %2i: rres = [%9.2e %9.2e %9.2e]\n',...
        iter,out.rres(end,:));
    if max(out.rres(end,:)) <= tol
        out.iter = iter;
        out.converged = true;
        break;
    end
end
end

function [ x,la,v] = my_TRS_solver( A,b,lamb,p,rho,tol,v_pre)
%% usgae: 
%  solve (x'*Q*x)/2 + c' * x
% with   x'*x <= 1;
%        Q = A + rho * p * p'
%        c = lamb * p -b
% Input:
%   (A,b,lamb,p,rho): input data
%                tol: tolerance for the generalized eigenvalue problem(GEP)
%              v_pre: previous solution to the GEP
% Output:
%                  x: solution to optimization
%                 la: associated multiplier
%                  v: solution to the GEP
%% parameter Setting
n = length(A);
if nargout < 7
    v_pre = rand(2*n,1);
end
%v_pre = rand(2*n,1);
c = lamb*p - b;
B = speye(n);
MM1 = [sparse(n,n) B;B sparse(n,n)];
%% Solving the eigenvalue problem
OPTS.tol = tol;
OPTS.isreal = true;
OPTS.p = 10;
OPTS.v0 = v_pre;
[v,~] = eigs(@(x)MM0timesx1(c,x,rho,A,p),2*n,-MM1,1,'lm',OPTS);
%% Derive the optimal solution
x = v(1:n);
normx = norm(x);         
x = x/normx; % in the easy case, this naive normalization improves accuracy
if x'*c>0, x = -x; end % take correct sign
la = -x'*A*x + x'*b;
end

function [y] = MM0timesx1(g,x,rho,A,p)
% compute MM0 * x with
% MM0 = [-I             A+rho*p*p';
%        A+rho*p*p'     -g*g'];
n = size(A,1); 
x1 = x(1:n); x2 = x(n+1:end);
y1 = -x1 + A*x2 + rho*2*(p'*x2)*p;
y2 = A*x1 -g*(g'*x2) +  rho*2*(p'*x1)*p;
y = [y1;y2];
end
