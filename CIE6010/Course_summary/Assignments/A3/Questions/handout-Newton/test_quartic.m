clear, close all

m = 50; n = m^2;
A = gallery('poisson',m);
u = (1:n)'*(4*pi/n); u = cos(u);
mu = 2;
x0 = u;
fprintf('\n m = %i, n = %i\n',m,n)

runcase = input('runcase (0 to 3) = ');
if isempty(runcase), runcase = 1; end

switch runcase
    case 0, func = 'yzQuartic';
        S = {'yzNewton','yzFdNewton'}; 
    case 1, func = 'yzQuartic';
        S = {'yzNewton','myNewton'};
    case 2, func = 'yzQuartic';
        S = {'yzFdNewton','myFdNewton'}; 
    case 3, func = 'myQuartic';
        S = {'myNewton','myFdNewton'}; 
end

tol = 1e-15;
F = zeros(2,1);
T = zeros(2,1);

for i = 1:2

    solver = S{i};
    fprintf(['\n--- Run ' solver ' ---\n'])
    t0 = tic;
    [x,iter] = eval([solver '(@' func ',x0,tol,20,A,u,mu);']);
    T(i) = toc(t0);
    [~,~,F(i)] = yzQuartic(x,A,u,mu);
    fprintf([solver ': iter %3i, tcpu = %6.2e\n'],iter,T(i))
    figure(i), plot(1:n,u,1:n,x); grid on;
    title([solver ' / ' func]); legend('u','x'); 
    pause(1)

end

fprintf('\nObj value:\n'); format long;  disp(F)
fprintf('\nTime used:\n'); format short; disp(T)