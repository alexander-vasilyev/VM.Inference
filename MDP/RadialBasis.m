function [ basis ] = RadialBasis( num,N,Radi,dim)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
ind=2*dim-1;
v=[1:1:ind];
v1= repmat(dim,1,ind);
v= v - v1;
v= repmat(v,1,ind);
v=vec2mat(v,ind);
v= v.*v;

w=[1:1:ind];
w1=repmat(dim,1,ind);
w=w-w1;
w= repmat(w,1,ind);
w= vec2mat(w,ind);
w=w.*w;

msum= w+v.';

msum= msum.^0.5;
cont= 0*ones(ind);
msum1= msum-cont;
msum0= msum;
% const= repmat(1, 128);
% R= const+ msum.*(pen)/200;
% R= R.^(-1);
msum(msum<Radi)=1;
msum(msum>(Radi-1))=0;
msum1(msum1<0)=0;

z=besselzero(num,N,1);
A= zeros(ind, ind, N);

for j= 1:N
    I=arrayfun(@(x) besselj(num,x*z(j)/(Radi)),msum1);
    An=I.*msum;
    if (sum(sum(An.*An)))>0
        A(:,:,j)=An/sqrt(sum(sum(An.*An)));
    end;
end;

basis=A;


