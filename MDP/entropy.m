function [ H ] = entropy( P ,D1,Dfft)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


[m,n] = size(P);
mb= 128;
nb= 128;

% output size 
mm = m + mb;
nn = n + nb;
% Pgpu=gpuArray(P);
% Cgpu=ifft2(fft2(Pgpu,mm,nn).*Dfft);
% padC_m = ceil((mb-1)./2);
% padC_n = ceil((nb-1)./2);
% Hgpu = Cgpu(padC_m+1:m+padC_m, padC_n+1:n+padC_n);
% H=gather(Hgpu);
% H=real(H);
% pad, multiply and transform back
C = ifft2(fft2(P,mm,nn).*Dfft);

%padding constants (for output of size == size(A))
padC_m = ceil((mb-1)./2);
padC_n = ceil((nb-1)./2);

% frequency-domain convolution result
H = C(padC_m+1:m+padC_m, padC_n+1:n+padC_n);




end

