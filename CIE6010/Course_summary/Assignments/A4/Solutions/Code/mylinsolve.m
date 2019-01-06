function [dx,dy,dz] = mylinsolve(A,rd,rp,rc,x,z)
% Usage: Solving LP Barrier Systems by Newton¡¯s Method
% Input:
%       A: m * n matrix
%      rd: n * 1 vector
%      rp: m * 1 vector
%      rc: n * 1 vector
%       x: n * 1 vector
%       z: n * 1 vector
%Output:
%      dx: n * 1 vector
%      dy: m * 1 vector
%      dz: n * 1 vector
n = length(rd);
d = x./z;
B = A * sparse(1:n,1:n,d) * A';
t1 = -x.*rd + rc;
t2 = A * (-t1./z) + rp;
dy = B \ t2;
dx = (t1 + x.*(A'*dy))./z;
dz = (rc - z.*dx)./x;
end

