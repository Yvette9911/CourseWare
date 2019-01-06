% solve random LP by PDIPM
% min c'*x, st.  Ax = b, x >= 0

clear
% initialize
tol = 1e-9;
maxit = 99;

r = input('[size m = 500*(1:r), n = 5*m] r = ');
if isempty(r), r = 1; end
RR = 1:r;

% fix random seed
if exist('rng','file')
    rng(0);
else
    s=RandStream('mt19937ar');
    RandStream.setDefaultStream(s); %#ok<*SETRS>
end

Time = zeros(1,3);
Iter = zeros(1,3);
Res  = zeros(1,3);
prt = 0; % turn off screen printout


% multiple runs
for k = 1:r
    
    m = 500*RR(k); n = 5*m;
    fprintf('\n====================================\n');
    fprintf('Trial %i: (m,n) = (%i,%i)\n',k,m,n);
    A = sprandn(m,n,0.005);
    b = sum(A,2); c = rand(n,1);
    p_res = @(x) norm(A*x-b)/(1+norm(b));
    d_res = @(y,z) norm(A'*y+z-c)/(1+norm(c));
    c_res = @(x,y) abs(c'*x-b'*y)/(1+abs(b'*y));
    
    % run Matlab linprog
    fprintf('  --- Matlab linprog ---\n');
    options = optimoptions(@linprog,'display','none',...
        'OptimalityTolerance',tol,...
        'Algorithm','interior-point');
    t0 = tic; [x,obj,~,output,lambda] = linprog...
        (c,[],[],A,b,zeros(n,1),[],[],options);
    t = toc(t0); toc(t0);
    iter = output.iterations;
    Time(1) = Time(1) + t;
    Iter(1) = Iter(1) + iter;
    y = -lambda.eqlin; z = lambda.lower;
    rp = p_res(x); rd = d_res(y,z); rc = c_res(x,y); 
    res = sum([rp,rd,rc]); Res(1) = Res(1) + res;
    fprintf('linprog  obj: %12.6e\n',obj);
    fprintf('p_res: %e\n',rp);
    fprintf('d_res: %e\n',rd);
    fprintf('c_res: %e\n',rc);
    fprintf('number of iter: %i\n\n',iter)
    
    
    % run yz_pdipm codes
    if exist('yz_pdipm','file')
        fprintf('  --- yz_pdipm ---\n');
        t0 = tic; [x,y,z,iter] = yz_pdipm(A,b,c,tol,maxit,prt);
        t = toc(t0); toc(t0);
        Time(2) = Time(2) + t;
        Iter(2) = Iter(2) + iter;
        rp = p_res(x); rd = d_res(y,z); rc = c_res(x,y);
        res = sum([rp,rd,rc]); Res(2) = Res(2) + res;
        fprintf('yz_pdipm obj: %12.6e\n',c'*x);
        fprintf('p_res: %e\n',rp);
        fprintf('d_res: %e\n',rd);
        fprintf('c_res: %e\n',rc);
        fprintf('number of iter: %i\n\n',iter)
    end
    
    % run my_pdipm codes
    if exist('my_pdipm','file')
        fprintf('  --- my_pdipm ---\n');
        t0 = tic; [x,y,z,iter] = my_pdipm(A,b,c,tol,maxit,prt);
        t = toc(t0); toc(t0);
        Time(3) = Time(3) + t;
        Iter(3) = Iter(3) + iter;
        rp = p_res(x); rd = d_res(y,z); rc = c_res(x,y);
        res = sum([rp,rd,rc]); Res(3) = Res(3) + res;
        fprintf('my_pdipm obj: %12.6e\n',c'*x);
        fprintf('p_res: %e\n',rp);
        fprintf('d_res: %e\n',rd);
        fprintf('c_res: %e\n',rc);
        fprintf('number of iter: %i\n\n',iter)
    end
    
end

fprintf('\n============ Statistics ============\n');
fprintf('  Solvers: linprog yzpdipm mypdipm\n')
fprintf('Average iter: %i\n',round(Iter/r))
fprintf('Average time: %f\n',Time/r)
fprintf('Average resi: %e\n',Res/r)
fprintf('====================================\n\n');
