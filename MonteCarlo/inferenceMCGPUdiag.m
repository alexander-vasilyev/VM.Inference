function [ meanMC,diagMC,X] = inferenceMCGPUdiag(A,D2,X,dim,grnsample,MCsteps)
% this function update the believe state according to observation vector O

dimm=2*dim;

DC=D2(((dim+1)-A(2)):(dimm-A(2)),((dim+1)-A(1)):(dimm-A(1)));
DC=DC(:);  

% mgc=max(DC);
% ind=DC>mgc/25;
% DCi=DC(ind);
% grnsamplei=grnsample(ind,:);
% cvmi=bsxfun(@times,DCi, grnsamplei);
% gmui=-0.5*DCi.*DCi;
% cvmi=bsxfun(@plus,gmui,cvmi);

 cvm=bsxfun(@times,DC,grnsample);
 gmu=-0.5*DC.*DC;
 cvm=bsxfun(@plus,gmu,cvm);

%X(ind,:)=X(ind,:) + cvmi; 

X = cvm+X;

sft=softmaxMCGPU(X);
mn=sum(sft,2)/MCsteps;
sft=bsxfun(@minus,sft,mn);
cn=sum(sft.^2,2)/MCsteps;

meanMC=vec2mat(gather(mn),dim);

cn=gather(cn);
 stdmat=vec2mat(cn,dim);
 diagMC=stdmat;


end