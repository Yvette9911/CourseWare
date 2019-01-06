% test L1 regularization for compressive sensing:
%        min ||Lx||_1, st. Ax = b  or
% min sum(sqrt((Lx).^2 + sigma)) + mu/2||Ax-b||^2

clear, close all
rng('default')

st = input('Solver (LP=0, GD=[1]): ');
if isempty(st), st = 1; end
switch st
    case 0,    S = {'myL1reg0','yzL1reg0'};
    otherwise, S = {'myL1reg1','yzL1reg1'};
end

%deg = input('Degree of D (1 or 2 [1:2]): ');
if ~exist('deg','var') || isempty(deg), deg = 1:2; end 
%m = input('Number of rows m [500] = ');
if  ~exist('m','var') || isempty(m), m = 500; end 

n = 2*m; k = ceil(m/5); A = randn(m,n);
fprintf('\n[m, n, k] = [%i, %i, %i]\n',m,n,k) 

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
eqs = ' ================= ';
warning off

h = length(deg);
E = zeros(2,h);
T = zeros(2,h);

for i = 1:2
    solver = S{i};
    if ~exist(solver,'file'), continue, end
    figure(i)
    for j = 1:h
        x0 = xo; p = deg(j);
        for l = 1:p-1, x0 = cumsum(x0); end
        b = A*x0;
        b = b + norm(b)*noise;
        t0 = tic;
        if st == 0
            x = eval([solver '(A,b,D{p})']);
        else
            [x,iter] = eval([solver '(A,b,D{p})']);
        end
        t1 = toc(t0); 
        relerr = norm(x-x0)/norm(x0);
        T(i,j) = t1; E(i,j) = relerr;
        if p==1, fprintf(['\n' eqs solver eqs '\n']); end
        fprintf('d = %1i: rel_err: %e  time: %f',p,relerr,t1)
        if st, fprintf('  iter: %i',iter); end
        fprintf('\n');
        subplot(h*100+10+j); plot(1:n,x0,'b-',1:n,x,'r.');
        axis auto; shg
        title([solver ' Results'])
    end
end

if ~isempty(T)
    re = E(1,:)./E(2,:);
    rt = T(1,:)./T(2,:);
    fprintf('\n ***** error and time ratios ******\n\n');
    disp([re; rt])
end

shg