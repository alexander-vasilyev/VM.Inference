function [ Hank ] = IniHankel2(  mean,CM,SM,dim,N,AN )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

ind=2*dim-1;
Hanki= repmat(zeros, [ind ind]);
for i=1:1
    for j= 1:N    
        Hanki= Hanki+ squeeze(CM(i,:,:,j)*mean((i-1)*(N)+j))+squeeze(SM(i,:,:,j)*mean(N*(AN+1)+(i-1)*(N)+j));
    end
end
Hanki(dim,dim)=mean(1);
Hank= Hanki;
end