
% ------------------------------------------
%    min x'*A*x/2 - b'*x + y'*A*y/2 - c'*y
%   s.t. (x'*x-1)/2=0, (y'*y-1)/2=0, x'*y=0
% ------------------------------------------

% main test script to run tests in comparison
%  with the instructor's code yz_qcqp_admm.p

fprintf('\n')
disp('parameter: n = problem size (default 3600)')
disp('parameter: s = scale b/c vs A (default 1)')
fprintf('\n')

clear
global n s Solvers

tol = 1.e-8;
Solvers = {'my_qcqp_admm','yz_qcqp_admm'};
rng('shuffle'); %rng('default')

n = input('n = ');
if isempty(n), n = 3600; end
n = ceil(sqrt(n))^2;
A = gallery('neumann',n);
A = -(A'*A);

b = randn(n,1); b = b/norm(b);
c = randn(n,1); c = c/norm(c);
Asiz = norm(A,'fro')/sqrt(n);
b = Asiz*b; c = Asiz*c;
    
s = input('s = ');
if isempty(s), s = 1; end
b = s*b; c = s*c;

run_test_final(A,b,c,tol);