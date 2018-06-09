function [ F ] = circmat( A )
%CIRCMAT Summary of this function goes here
%   Detailed explanation goes here
nl=size(A);
n=nl(1);
a = reshape(A,n^2,1);
E0 = eye(n);
F = zeros(n^2);
f1 = zeros(n^2,1);
for k = 1:n^2
% shifts
i = mod(k-1, n);
j = floor((k-1)/n);
% shift operators
Ei = circshift(E0, [i,0]);
Ej = circshift(E0, [j,0]);
% first row of Dij = kron(Ej,Ei)
d1 = kron(Ej(1,:),ones(1,n)) .* repmat(Ei(1,:), 1,n);
% entry F(k,1)
f1(k) = d1 * a;
% kronecker construction
% Dij = kron(Ej,Ei);
% F(k,:) = Dij * a;
% direct construction
F(k,:) = reshape(circshift(A, [i,j]), 1, n^2);
end

end

