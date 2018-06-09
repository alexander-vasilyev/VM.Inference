function [ convres ] = toepcnv( ftoep,meanMC,dim)
%TOEPCNV Summary of this function goes here
%   Detailed explanation goes here
dimm=2*dim-1;
meanf=flipud(meanMC);
meanf=transpose(meanf);
meanf=meanf(:);

res=ftoep*meanf;

res=vec2mat(res,dimm);
convres=flipud(res);


end

