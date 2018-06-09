function [ DMC ] = decisionMC(Imean,Icov,MCsteps,gr,dims,rnsample)
% this function update the believe state according to observation vector O

Nmc=MCsteps;
mu=flipud(Imean);
mu=transpose(mu);
mu=mu(:);

cvm=Icov*rnsample;
%cvm=Icov*randn(dims*dims,Nmc);
bgmu=repmat(mu,1,MCsteps);
X=bgmu+cvm;
%   sft=softmaxMC(X/50);
%   mn=mean(sft,2);
mn=hardmaxMC(X,dims);

mn=vec2mat(mn,dims);
mn=flipud(mn);

%DMC=smooth2a(mn,1,1);
DMC=imgaussfilt(mn,1);
if gr
figure
 colormap('hot');
 imagesc(DMC);
 set(gca,'YDir','normal')
 colorbar;
 title('average probability of decision');
 drawnow;
 pause(1);
end;


end