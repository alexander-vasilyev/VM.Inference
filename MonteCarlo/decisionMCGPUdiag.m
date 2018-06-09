function [ DMC ] = decisionMCGPUdiag(Imean,Icov,gr,dim,filt,A,grnsample,MCsteps)
% this function update the believe state according to observation vector O

%  mgc=max(Icov);
%  ind=Icov>mgc/25;
%  Icovi=Icov(ind);
%  grnsamplei=grnsample(ind,:);
%  cvmi=bsxfun(@times,Icovi, grnsamplei);
%  cvm=0*grnsample;
%  cvm(ind,:)=cvmi;

cvm=bsxfun(@times,Icov, grnsample);

X=bsxfun(@plus,Imean,0.1*cvm);

mn=hardmaxMC(X,dim);
%mn=softmaxMCGPU(X);
%mn=sum(mn,2)/MCsteps;
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
DMC=ApplyGaussianBlur(mn,filt,filt);
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