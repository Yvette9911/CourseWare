function test_ipm2(ii)
%
% test on the inpainting problem
%

% change the following path to suit your case
if nargin < 1, ii = 1; end
eval(['load data' int2str(ii) '_inpaint']);

% display information
fsnr = @(Sig,Noi) 20*log10(norm(Sig(:))/norm(Noi(:)));
SNR1 = fsnr(Img1,Img1-Img0);
fprintf('--- Image cameraman: %.2f%% pixels missing\n',...
       (1-numel(Omega)/numel(Img0))*100)
fprintf('--- Initial SNR = %g\n\n',SNR1)
xstr(1,:) = 'Recovery by linprog';
xstr(2,:) = 'Recovery by yzpdipm';
xstr(3,:) = 'Recovery by mypdipm';
close all

Solvers = 1:3;
for k = Solvers % different solvers
    
    % recover image
    tic; fprintf([xstr(k,:) ' ...\n']);
    Img2 = Inpaint(Img1,Omega,k); toc
    SNR2 = fsnr(Img2,Img2-Img0);
    strL = 'Left: Original. \t';
    strM = 'Middle: Contaminated (SNR: %.2f). \t';
    strR = 'Right: Recovered (SNR: %.2f)';
    str = sprintf([strL strM strR],SNR1,SNR2);
    
    % display images
    warning('off','images:initSize:adjustingMag');
    figure(k); imshow([Img0 Img1 Img2],[])
    h1 = title(str); h2 = xlabel(xstr(k,:));
    set(h1,'fontsize',16,'FontWeight','bold')
    set(h2,'fontsize',18,'FontWeight','bold')
    drawnow; shg
    fprintf('Recovered SNR = %g\n\n',SNR2);
    
end

end

%% call pdipm to do recovery
function X = Inpaint(X1,Omega,k)
% construct LP data
m = numel(Omega);
n = size(X1,1);
n2 = n^2;

x = X1(:);
y = x(Omega);

e = ones(n,1); I = speye(n);
D = spdiags([e -e],0:1,n-1,n);
D = [kron(I,D); kron(D,I)];
S = speye(n2);
S = S(Omega,:);

% solve LP
% min u + v, st. Dx - u + v = 0, Sx = y, x,u,v >= 0
nD = size(D,1);
A = [D -speye(nD) speye(nD);
    S   sparse(m,2*nD)];
b = [zeros(nD,1); y];
c = [zeros(n2,1); ones(2*nD,1)];
tol = 1e-6; maxit = 99;
prt = 1; % turn on printout

switch k
    case 1
        options = optimoptions(@linprog,'display','none',...
            'OptimalityTolerance',tol,...
            'Algorithm','interior-point');
        [x,~,~,output] = ...
            linprog(c,[],[],A,b,zeros(n2+2*nD,1),[],[],options);
        iter = output.iterations;
    case 2 
        [x,~,~,iter] = yz_pdipm(A,b,c,tol,maxit,prt);
    case 3 
        [x,~,~,iter] = my_pdipm(A,b,c,tol,maxit,prt);
end

fprintf('number of iterations: %i\n',iter)
X = reshape(x(1:n2),n,n);
end
