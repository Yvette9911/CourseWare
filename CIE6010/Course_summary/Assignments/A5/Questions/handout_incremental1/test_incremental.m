% test the behavior of incremental gradient method on:
%         min 1/2*||Ax - b||^2 

clear, close all

n = 100;                    % vector size
x0 = zeros(n,1);            % initial guess
xo = sin(3*pi/n*(1:n)');    % original 
nrmxo = norm(xo);

S = {'yzIncremental','myIncremental'};
density = 5e-2; rcond = 5e-4;
tol = 1e-4; maxit = 10000; 
M = [20 40]*n;
sigma = .1;

for i = 1:2
    
    solver = S{i};
    if ~exist(solver,'file'), continue; end

    for j = 1:numel(M)
        
        m = M(j);
        rng(0)
        A = sprandn(m,n,density,rcond);        
        b = A*xo; noise = randn(m,1);
        b = b + sigma*norm(b)/norm(noise)*noise;
        xs = (A'*A)\(A'*b);
        rerr1 = norm(xs-xo)/nrmxo;
        fprintf(['\nsolver = ' solver '  '])
        size_str = sprintf('[m, n] = [%i, %i]',m,n);
        fprintf([size_str '\n']);
            
        tic, x = feval(solver,A,b,x0,tol,maxit); toc        
        rerr2 = norm(x-xo)/nrmxo;
        fprintf('relative errors = [%f, %f]\n',rerr1,rerr2);
        figure((i-1)*2+j);
        H = plot(1:n,xo,1:n,xs,'ks:',1:n,x,'ro:');
        set(H,'linewidth',2)
        legend({'original','direct sol.','incremental'},...
            'Location','southeast','Fontsize',14);
        title(solver,'fontsize',14)
        xlabel(size_str,'fontsize',14)
        shg
        
    end
    fprintf('\n')
end