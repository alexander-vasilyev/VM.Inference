function [ cvm ] = toepcovmat( ftoep,ftoeptr,diagMC,indmat,dim)
%TOEPCNV Summary of this function goes here
%   Detailed explanation goes here
%  dimm=2*dim-1;
%  diagf=flipud(diagMC);
%  diagf=transpose(diagf);
% %  diagMC=diagMC';
%   diagf=diagf(:);


%tempr=bsxfun(@times,diagMC,ftoeptr);
tempr=diagMC*ftoeptr;

cvm=ftoep*tempr;

cvm=cvm(indmat,indmat);

end



