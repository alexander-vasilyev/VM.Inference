function [ Dfft ] = Dfft(P, D1)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

D11= D1(65:193,65:193);
[m,n] = size(P);
[mb,nb] = size(D11); 
% output size 
mm = m + mb - 1;
nn = n + nb - 1;

Dfft= fft2(D11,mm,nn);





end
