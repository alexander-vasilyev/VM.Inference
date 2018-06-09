function [ meanMC,diagMC,X] = inferenceMCGPUfull(A,D2,X,MCsteps,dim)
% this function update the believe state according to observation vector O



ind=2*dim;
DC=D2(((dim+1)-A(2)):(ind-A(2)),((dim+1)-A(1)):(ind-A(1)));
gcovvec=DC(:);


grnsample=randn(dim*dim,MCsteps,'gpuArray');
cvm=bsxfun(@times,gcovvec, grnsample);
gmu=-0.5*gcovvec.*gcovvec;
cvm=bsxfun(@plus,gmu,cvm);

X = cvm+X;



sft=softmaxMCGPU(X);
%sft=gather(sft);
mn=sum(sft,2)/MCsteps;
  sft=bsxfun(@minus,sft,mn);
  %cn=sum(sft.^2,2)/MCsteps;
cn=cov(sft');
%cn=var(sft,0,2);
%mn=gather(mn);
%mn=vec2mat(mn,dim);
%meanMC=mn';
meanMC=mn;

cn=gather(cn);
 
 diagMC=cn;
%diagMC=cn; 


end