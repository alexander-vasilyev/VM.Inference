function [ CM,SM ] = CircBasis( dim ,N,AN,Radi)
%RADIALMASK Summary of this function goes here
%   Detailed explanation goes here
ind=2*dim-1;

v=[1:1:ind];
v1= repmat(dim,1,ind);
v= v - v1;
v= repmat(v,1,ind);
v=vec2mat(v,ind);
v=v';

w=[1:1:ind];
w1=repmat(dim,1,ind);
w=w-w1;
w= repmat(w,1,ind);
w= vec2mat(w,ind);

CM=repmat(1, [AN+1 ind ind N]);
SM=repmat(1, [AN+1 ind ind N]);
ONEM=ones(ind,ind);
BASISS=repmat(1,[AN+1 ind ind N]);
for k=1:(AN+1)
    BASISS(k, :, :, :)=RadialBasis(k-1,N,Radi,dim);
end;

for k=1:(AN+1)
    for l=1:N
        for i=1:ind
            for j=1:ind
                atn=atan2(v(i,j),w(i,j));               
                if atn<0
                    atn=atn+2*pi;
                end;
                CM(k,i,j,l)=BASISS(k,i,j,l)*cos((k-1)*atn);
                SM(k,i,j,l)=BASISS(k,i,j,l)*sin((k-1)*atn);
            end;  
        end;
    end;
end;
end