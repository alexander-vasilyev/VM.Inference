function [ DMC ] = decisionMCGPUfull(Imean,Icov,MCsteps,gr,dim,filt,A)
% this function update the believe state according to observation vector O


% mu=flipud(Imean);
% mu=transpose(mu);
% mu=mu(:);
cvm=randn(dim*dim,MCsteps,'gpuArray');

cvm=Icov*cvm;

X=bsxfun(@plus,Imean,cvm);

%mn=hardmaxMC(X,dim);
  mn=softmaxMC(X);
  mn=sum(mn,2)/MCsteps;

mn=gather(mn);
% 
mn=vec2mat(mn,dim);
mn=mn';
%mn=flipud(mn);


smrad=ceil(dim/16);
% 
if filt==0
DMC=mn;
else
DMC=imgaussfilt(mn,filt);
end;
DMC=DMC/sum(sum(DMC));

if gr
 figure;
 colormap('hot');
 str='probability of decision';   
 imagesc(DMC);   
 set(gca,'YDir','normal')
 title(str); 
 hold on;
 plot(A(1),A(2),'b*');
 pause(2);
end;


end