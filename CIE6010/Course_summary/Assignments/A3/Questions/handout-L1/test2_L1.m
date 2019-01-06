% test L1 regularization for compressive sensing:
%        min ||Lx||_1, st. Ax = b  or
% min sum(sqrt((Lx).^2 + sigma)) + mu/2||Ax-b||^2

clear, close all
rng(2)

global sigma mu alpha
sigma = 1e-2; mu = 1e-1;

runcase = input('runcase (0 to 4) = ');
if isempty(runcase), runcase = 0; end

switch runcase
    case 0, S = {'yzL1reg2a','yzL1reg2b','yzL1reg2c'}; 
        m = 500; deg = 1:2;
    case 1, S = {'myL1reg2b','myL1reg2c'};
        m = 200; deg = 1:2; 
    case 2, S = {'myL1reg2a','myL1reg2c'};
        m = 400; deg = 1:2;
    case 3, S = {'yzL1reg2b','myL1reg2b'};
        m = 600; deg = 1;
    case 4, S = {'yzL1reg2c','myL1reg2c'};
        m = 1000; deg = 1:2;
    otherwise, S = {'yzL1reg2c'}; 
        m = 1500; deg = 1;
end

n = 2*m; k = ceil(m/5); A = randn(m,n);
fprintf('\n[m, n, k] = [%i, %i, %i]\n',m,n,k)

Lip = 1 + mu*norm(A*A');
alpha = 1 / Lip; 
fprintf('[sigma mu alpha] = [%.2e %.2e %.2e]\n',sigma,mu,alpha)

snr = 50;
noise = randn(m,1);
noise = 10^(-snr/20)/norm(noise)*noise;

range = 50;
rp = randperm(n);
xo = zeros(n,1);
xo(rp(1:k))= range*randn(k,1);

D = cell(3,1);
e = ones(n,1);
D{1} = spdiags(e,0,n,n);
D{2} = spdiags([-e e],-1:0,n-1,n);
D{3} = spdiags([-e 2*e -e],-1:1,n-2,n);
eqs = ' ===================== ';
warning off

nS = length(S);
h = length(deg);
E = zeros(nS,h);
T = zeros(nS,h);

for i = 1:nS
    solver = S{i};
    if ~exist(solver,'file'), continue, end
    figure(i)
    for j = 1:h
        x0 = xo; p = deg(j);
        for l = 1:p-1, x0 = cumsum(x0); end
        b = A*x0;
        b = b + norm(b)*noise;
        t0 = tic;
        [x,iter] = eval([solver '(A,b,D{p})']);
        t1 = toc(t0); 
        relerr = norm(x-x0)/norm(x0);
        T(i,j) = t1; E(i,j) = relerr;
        if p==1, fprintf(['\n' eqs solver eqs '\n']); end
        fprintf('p = %1i: rel_err: %e  time: %9.4f  iter: %i\n',...
            p,relerr,t1,iter)
        subplot(h*100+10+j); plot(1:n,x0,'b-',1:n,x,'r.');
        axis auto; shg
        title([solver ' Results'])
    end
end

fprintf('\n ***** Error ******\n\n');
disp(E)
fprintf('\n ***** Time ******\n\n');
disp(T)