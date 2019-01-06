% Gauss-Newton/Newton method for solving
% the phrase retrieval problem:
%    min (1/2)*norm(|Hx|^2 - d)^2
% where H = A+iB is complex and d = |Hx^*|^2.

clear, close all
rng('default')
%rng('shuffle')

tol = 1e-11; 
maxit = 200;

n = input('(return for default) n = ');
if isempty(n), n = 500; end; m = ceil(5*n);
fprintf('Phase: n = %i, m = %i\n\n',n,m)

pm = randperm(m);
pn = sort(pm(1:n));

t = (1:m)'/m; 
xo = exp(-t.^2).*cos(pi*t).^2;
xs = xo(pn);

I = speye(m);
H = fft(full(I(:,pn)));
A = real(H);
B = imag(H);
d = abs(H*xs).^2;
dnrm2h = norm(d)^2/2;

sigma = 3e-2; noise = randn(m,1);
d = d + sigma*norm(d)/norm(noise)*noise;
dnrm2 = 0.5*norm(d)^2;
x0 = randn(n,1);

solver{1} = 'yzphase';
solver{2} = 'myphase';
In = '(A,B,d,x0,tol,maxit,Itmin(i));';
Itmin = [maxit, 3];

for s = 1:2
    if exist(solver{s},'file')
        
        t0 = tic;
        for i = 1:2
            t1 = tic;
            [x,hcrit,iter] = eval([solver{s} In]);
            rerr = norm(abs(x)-abs(xs))/norm(xs);
            fprintf('--- %s: iter %3i relerr = %.3e  time %g ---\n\n',...
                solver{s},iter,rerr,toc(t1))
            figure(s); subplot(121) 
            h = semilogy(1:iter,hcrit,'-o');
            set(h,'linewidth',2); 
            axis square; grid on
            title([solver{s} ' Iter history'],'fontsize',14);
            xlabel('Iteration'); ylabel('scaled g-norm');
            grid on; drawnow; shg
            if i == 1, hold; else, hold off; end
        end
        t0 = toc(t0); 
        
        subplot(122); if i == 1, hold; end
        h = plot(1:n,xs,'-',1:n,abs(x),'.');
        set(h,'linewidth',2); 
        str = sprintf(' (t=%.2fs)',t0);
        title([solver{s} ' solutions' str],'fontsize',14); 
        axis square; grid on
        %xlabel(''); ylabel('');
    end
    drawnow; shg
    
end