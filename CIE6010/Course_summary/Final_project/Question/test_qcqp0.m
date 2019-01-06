% ------------------------------------------
%    min x'*A*x/2 - b'*x + y'*A*y/2 - c'*y
%   s.t. (x'*x-1)/2=0, (y'*y-1)/2=0, x'*y=0
% ------------------------------------------

% test script on small problems in comparison with 
% instructor's code calling the Matlab fmincon

fprintf('\n')
disp('parameter: n = problem size (default 400)')
disp('parameter: s = scale b/c vs A (default 1)')
fprintf('\n')

clear
global n s Solvers

tol = 1.e-6;
Solvers = {'yz_qcqp_fmin','my_qcqp_admm'};

n = input('n = ');
if isempty(n), n = 400; end
n = ceil(sqrt(n))^2;
A = gallery('neumann',n);
A = -(A'*A); 

rng('shuffle'); rng('default')
b = randn(n,1); b = b/norm(b);
c = randn(n,1); c = c/norm(c);
Asiz = norm(A,'fro')/sqrt(n);
b = Asiz*b; c = Asiz*c;

s = 1;
%s = input('s = ');
%if isempty(s), s = 1; end
b = s*b; c = s*c;

Out = run_test(A,b,c,tol);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Out = run_test(A,b,c,tol)
global n s Solvers

I = speye(n);
X = zeros(n,2);
Y = zeros(n,2);
Obj = zeros(2,1);
nsolve = 0; fn = 1;

fobj = @(x,y).5*(x'*A*x + y'*A*y) - b'*x - c'*y;

for j = 1:length(Solvers)
    
    solver = Solvers{j};
    if ~exist(solver,'file'), continue; end
    fprintf(['\n---- solver: ' solver '  n = %i  s = %.2f ---- \n'],n,s);
    
    tic, [x,y,lamb,out] = feval(solver,A,b,c,tol); toc, toc, toc
    X(:,j) = x; Y(:,j) = y; Obj(j) = fobj(x,y); 
    nsolve = nsolve + 1;
    
    DxL = A*x - b + lamb(1)*x + lamb(3)*y;
    DyL = A*y - c + lamb(2)*y + lamb(3)*x;
    Fv = [(x'*x-1)/2; (y'*y-1)/2; x'*y];
    
    fprintf(['\nsolver: ' solver '  n = %i  s = %.2f\n'],n,s);
    fprintf('Optimality: [%6.2e %6.2e %6.2e]\n',...
        norm(DxL)/(1+norm(b)),norm(DyL)/(1+norm(c)),norm(Fv))
    fprintf('Convergence = %i at iter %i\n\n',...
        out.converged,out.iter)
    
    if isfield(out,'rres')
        figure(fn); fn = fn + 1;
        semilogy(out.rres,'linewidth',2); grid on; shg
        legend('DxL(x,y,lamb)','DyL(x,y,lamb)','(x,y) Feasibility')
        xlabel('iter'), ylabel('residuals')
        title('Iteration history of residuals')
    end
    
    if isfield(out,'inner')
        figure(fn); fn = fn + 1;
        plot(out.inner,'linewidth',2); grid on; shg
        legend('x-subprob','y-subprob')
        xlabel('iter'), ylabel('No. of inners')
        title('History of inner iteration numbers')
    end
    
    if n <= 100000 && strcmp(solver,'yz_qcqp_admm')
        tic
        H = [A+lamb(1)*I lamb(3)*I; lamb(3)*I A+lamb(2)*I];
        lamin = eigs(H,1,'smallestreal');
        fprintf('Checking: 2nd-order sufficiency: %i\n',lamin>0)
        toc
    end
    
end

if nsolve > 1
    err_x = norm(X(:,1:2)*[1; -1]);
    err_y = norm(Y(:,1:2)*[1; -1]);
    fprintf('x and y errors: [%6.2e %6.2e]\n',err_x,err_y)
end

fprintf('objective values:\n')
format long, disp(Obj), format short
Out.X = X; Out.Y = Y; Out.Obj = Obj;

end