function [ RF ] = Radialupdate2( speed,dim,symm,staypenalty )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
ind=2*dim-1;
v=[1:1:ind];
v1= repmat(dim,1,ind);
v= v - v1;
v= repmat(v,1,ind);
v=vec2mat(v,ind);
v= v.*v;
v4=v.*v;
v4=v4.';

w=[1:1:ind];
w1=repmat(dim,1,ind);
w=w-w1;
w= repmat(w,1,ind);
w= vec2mat(w,ind);
w=w.*w;
w4=w.*w;

msum0=w+v.'+0.001;
if  ~symm
    msum= w+v.'-0.52*v4./msum0;
else
    msum=w+v.';
end;
%msum0=msum;



msum0=msum0.^(1/2);

msum=msum.^(1/2);

%no saccades below 1 deg
% msum0(msum0<15)= 0;
% msum0(msum0>0)=1;
%msum0=arrayfun(@(x) 1-exp(-10*x),msum0);



msum1=200+speed*msum;
msum1(dim,dim)=staypenalty;
msum1= msum1.^(-1);

% if shsac
%     msum0=arrayfun(@(x) 1-exp(-10*x),msum0);
%     msum1= msum1.*msum0;
% % end;

RF=msum1;
end