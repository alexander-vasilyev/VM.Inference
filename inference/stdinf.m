function [ convres ] = stdinf(ftoeptr,diagMC,dim,indmat)
%TOEPCNV Summary of this function goes here
%   Detailed explanation goes here


tempr=bsxfun(@times,diagMC,ftoeptr);
convres=tempr(:,indmat);
end
