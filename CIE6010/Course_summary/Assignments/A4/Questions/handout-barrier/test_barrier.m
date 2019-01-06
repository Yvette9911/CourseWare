% test linear solvers for LP barrier systems

% parameters
p = input('[size m = 500*p, n = 5*m] p = ');
if isempty(p), p = 1; end

m = 500*p; n = 5*m;
fprintf('(m,n) = (%i,%i)\n',m,n);
rng('default');
A = sprandn(m,n,0.01);
b = sum(A,2);
c = rand(n,1);
x0 = ones(n,1)*n;
y0 = zeros(m,1);
z0 = ones(n,1)*n;
sigma = .1;
maxit = 99;
tol = 5e-8;

S = {'mylinsolve','yzlinsolve'};
T = zeros(1,2);

for i = 1:2
    
    solver = S{i};
    x = x0; y = y0; z = z0;
    if ~exist(solver,'file'), continue; end
    fprintf('\n===============');
    fprintf(' using %s ',solver)
    fprintf('===============\n');
    tic
    
    for k = 1:maxit
        
        agap = (x'*z)/n;
        mu = sigma*agap;
        
        rd = c - A'*y - z;
        rp = b - A*x;
        rc = mu - x.*z;
        
        errd = norm(rd);
        errp = norm(rp);
        errc = norm(rc);
        
        fprintf('iter %2i: errors = [%6.2e, %6.2e, %6.2e]\n',...
            k,errd,errp,errc)
        maxerr = max([errd errp errc]);
        if maxerr < tol; break; end
        
        [dx,dy,dz] = feval(solver,A,rd,rp,rc,x,z);
        ap = -1/min(min(dx./x),-1);
        ad = -1/min(min(dz./z),-1);
        t = max(.99,1-maxerr)*min(ap,ad);
        
        x = x + t*dx;
        y = y + t*dy;
        z = z + t*dz;
        
    end
    T(i) = toc;
    
end

fprintf('\nUsing %s, time = %f\n',S{1},T(1));
fprintf('Using %s, time = %f\n\n',S{2},T(2));