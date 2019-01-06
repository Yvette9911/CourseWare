function [x,y,z,iter] = my_pdipm(A,b,c,tol,maxit,prt)
% Input:
%           A = constraint coefficient matrix
%           b = constraint right-hand side vector
%           c = objective coefficient vector
%         tol = tolerance
%       maxit = maximum number of iterations allowed
%         prt = switch for screen printout (1 on, 0 off)
% Output:
%           x = computed primal solution
%           y = computed dual solution
%           z = computed dual slacks
%        iter = iteration counter
%% parameters setting
%p = symamd(abs(A)*abs(A)');     
[m,n]=size(A);
p = symamd(abs(A)*abs(A)');%colamd(A');%symamd(A*A');%colamd(A');
%% Initialize
bigMfac = 100;
bigM = max(max(abs(A)));
bigM = max([norm(b,Inf), norm(c,Inf), bigM]);
x = bigMfac*bigM * ones(n,1); %bigMfac*bigM * ones(n,1);    
y = zeros(m,1);     
z = x;
%bc = 1 + max(norm(b),norm(c));
bc = 1+[norm(b);norm(c);0];
for iter = 1:maxit
    %% initial setting
    rd = -c + A'*y + z;
    rp = -b + A*x;
    rc = x.*z;
    gap = mean(rc);
    %% Stopping criteria
    bc(3) = abs(b'*y)+1;
    residual = sum([norm(rp);norm(rd);norm(rc)]./bc);
    if residual <= tol, break;end;
    switch prt
        case 1
            fprintf('iter %2i: [primal dual gap] = [%9.2e  %9.2e  %9.2e]\n',iter,[norm(rp);norm(rd);norm(rc)]./bc);
    end
    %% formulate the linear systems
    d = min(5.e+15,x./z);
    M = A* sparse(1:n,1:n,d) *A';   R = chol(M(p,p));%R = chol(M(p,p),'lower','vector');
    %% Predictor step: Solve for dy
    t1 = x.*rd - rc;
    t2 = -(rp+A*(t1./z));
    dy1 = zeros(m,1);
    dy1(p) = R\(R'\t2(p));
    %% Predictor step: Solve for dz, dx by back substitutions
    dx1 = (t1 + x.*(A'*dy1))./z;
    dz1 = (-rc - z.* dx1)./x;
    %% Corrector Step
    ap = -1/min(min(dx1./x),-1);
    ad = -1/min(min(dz1./z),-1);
    % centering parameter step
    mun = ((x + ap * dx1)' * (z + ad * dz1))/n;
    sigma = (mun/gap)^3;
    mu = gap * min(.2,sigma);
    rd = 0; rp = 0; rc = -mu + dx1.*dz1;
    t1 = x.*rd - rc;
    t2 = -(rp+A*(t1./z));
    dy2 = zeros(m,1);
    dy2(p) = R\(R'\t2(p));
    dx2 = (t1 + x.*(A'*dy2))./z;
    dz2 = (-rc - z.* dx2)./x;
    %% Combined Step
    dx = dx1+dx2;   dy = dy1 + dy2; dz = dz1+dz2;
    tau = max(.9995,1-10*gap);
    ap = -1/min(min(dx./x),-1);
    ad = -1/min(min(dz./z),-1);
    ap = min(tau * ap,1);
    ad = min(tau * ad,1);
    x = x + ap * dx;
    z = z + ad * dz;
    y = y + ad * dy;
end
x=full(x); z=full(z); y=full(y);
iter = iter-1;
end
