% minimum degree ordering and sparse cholesky
load ship04s;               % load a sparse matrix A
B = A*A' + speye(402);      % Guarantee positive definiteness
p = symamd(B);   % SYMmetric Approximate Minimum Degree ordering
% spy, spy, spy and spy
subplot(221); spy(B);
subplot(222); spy(B(p,p));
subplot(223); spy(chol(B));
subplot(224); spy(chol(B(p,p))); title('chol(B(p,p))');
