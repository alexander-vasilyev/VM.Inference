function [ conc ] = concentration( mixing,M,N)
concar=zeros(1,N);
for i=1:N
    [concar(i), m]=max(mixing(i,:));
end;
conc=mean(concar);
end