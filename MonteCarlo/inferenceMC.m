function [ meanMC,diagMC,X] = inferenceMC(DC,X,MCsteps,gr,dims,rnsample)
% this function update the believe state according to observation vector O
X0=X;
Nmc=MCsteps;
covvec=DC(:);
mu=-0.5*covvec.*covvec;

cvm=bsxfun(@times,covvec,rnsample);
cvm=bsxfun(@plus,mu,cvm);
%cvm=covmat*randn(dims*dims,Nmc);
X = cvm+X0;
%X = 0*repmat(mu,1,Nmc) +cvm+0*X0;


sft=softmaxMC(X);
mn0=mean(sft,2);
mn=vec2mat(mn0,dims);
mn=mn';

cn=var(sft,0,2);
if gr
figure
 colormap('hot');
 imagesc(mn);
 set(gca,'YDir','normal')
 colorbar;
 title('average probability of target location');
 drawnow;
 pause(2);
end;
 meanMC=mn;
 
 
 stdmat=vec2mat(cn,dims);
 stdmat=stdmat';
 diagMC=stdmat;
if gr 
 figure
 colormap('hot');
 imagesc(stdmat);
 set(gca,'YDir','normal')
 colorbar;
 title('variance of probability of target location');
 drawnow;
 pause(2);
end;
end