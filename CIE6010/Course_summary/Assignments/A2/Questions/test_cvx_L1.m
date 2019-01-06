% test L1 regularization for compressive sensing:
%        min ||Lx||_1, st. Ax = b
% using CVX

clear, close all
rng(2)

m = 150*2;
n = 2*m;
r = 4;

k = ceil(m/r);
A = randn(m,n);

snr = 50;
noise = randn(m,1);
noise = 10^(-snr/20)/norm(noise)*noise;

range = 50;
p = randperm(n);
xo = zeros(n,1);
xo(p(1:k))= range*randn(k,1);

D = cell(3,1);
e = ones(n,1);
D{1} = spdiags(e,0,n,n);
D{2} = spdiags([-e e],-1:0,n-1,n);
D{3} = spdiags([-e 2*e -e],-1:1,n-2,n);
eqs = '=================';
warning off

x0 = xo; figure(3)
for p = 1:3
    b = A*x0;
    b = b + norm(b)*noise;
    L = D{p};
    t0 = tic;
    
    cvx_begin quiet
    variable x(n)
    minimize( norm( L * x, 1 ) )
    subject to
    A * x == b; %#ok<EQEFF>
    cvx_end
    
    t1 = toc(t0);
    relerr = norm(x-x0)/norm(x0);
    if p==1, fprintf(['\n   ' eqs ' CVX ' eqs '\n']); end
    fprintf('deg = %1i: rel_err: %e  time: %f\n',p,relerr,t1)
    subplot(310+p); plot(1:n,x0,'b-',1:n,x,'r.');
    axis auto; shg
    title(sprintf('CVX Result: deg = %1i',p))
    x0 = cumsum(x0);
end