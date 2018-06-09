addpath('MDP');
addpath('convolutions');
addpath('filter');
addpath('inference');
addpath('interface');
addpath('MonteCarlo');
addpath('classification');
%number of cells
dim=64;
dimm=2*dim-1;
circ=false;
stats=true;
alsym=true;
%dimensionality of Kernel
N=1;
AN=2;
%number of modalities
MCsteps=512;

M=1;
%display steps
gr=false;
%display maps
mgr=true;
%sim eye movements
seye=false;

%size of validation set
valid=0.0;

devi= ones(2*N*(AN+1),M);

filt=1;
smdst=1;
gamma=0.00;

%number of epochs
epnum=length(trajcl);
eplength=1;
 
% 
if (gr==false) 
      delete(gcp('nocreate'));
       reset(gpuDevice(1));
       parpool('local',11,'SpmdEnabled',false);
end;

imbord=dim;

Obmat=repmat(zeros, [5 epnum]);

vObdist=repmat(zeros, [5 epnum]);
%index extraction in convolutions
indmat=[];
for j=1:dim*dim
    jblock=floor((j-0.5)/dim);
    indmat(end+1)=j+jblock*(dim-1);
end;
lbound=round(dim/2);
lm=1+dimm*(lbound-1)+lbound;
%random block size
blockdiv=24;
regular=10^(-8);


P=rand(dim);
%initial probability
if circ
    P0 = InitialP(dim);
else 
    P0 = repmat(1/(dim*dim) , [dim dim]);
end
 

prseq={};

for pen=1:length(trajcl)
disp(pen);
%select sequence
Xtraj=trajcl{pen}(1,:);
Ytraj=trajcl{pen}(2,:);

rng(1,'simdTwister');

%save coefficient for different epochs

probseq=cell(1,length(infval));
%start evaluation for selected sequence of eye-movements 

parfor repoch = 1:length(infval)
    disp(repoch);
    lpsum= zeros(1,eplength);
           
    eprob=0;
    %sum of log-probability for sequence     
    lprob=0;

    D2=infval{repoch};
    D1=D2.*D2;
    HFun1=rot90(rot90(D1));
    HFun1=HFun1(round(dim/2):(round(dim/2)+dim-1),round(dim/2):(round(dim/2)+dim-1));
    [Pfn,Hfn, fsize] = padarrays(P, HFun1,'same');
    visdft1=fft2(Hfn);
    visdfts1=fft2(Hfn.*Hfn);

    %evaluate log-probabilities for two maps

    A=[Xtraj(2),Ytraj(2)];

    OX= gpuArray(repmat(zeros,[dim*dim,MCsteps]));

    visdft=visdft1;
    visdfts=visdfts1;
    HFun=HFun1;
    D2=gpuArray(D2);

    j=2;
    limitr=4000;
    while ( j<(length(Xtraj)-2))&&(j<limitr)                        
    An=[Xtraj(j+1),Ytraj(j+1)]; 

    if (An(1)*An(1)+An(2)*An(2))>0

        grnsample=randn(dim*dim,MCsteps,'gpuArray');

        [ meanMC,diagMC,OX ] = inferenceMCGPUdiag(A,D2,OX,dim,grnsample,MCsteps); 

        % expected mean of information gain at each location

        Imean=conv_fft2(meanMC',HFun,visdft,'same',dim,fsize);

        H= conv_fft2(diagMC',HFun,visdfts,'same',dim,fsize);

        % variance of information gain at each location

        x0=A(1); 
        y0=A(2);

        %select next location
        A=[Xtraj(j+1),Ytraj(j+1)]; 
        j=j+1;

        Imean=gpuArray(Imean(:));
        Infcov=gpuArray(sqrt(H(:)));

        DMC=decisionMCGPUdiag(Imean,Infcov,gr,dim,filt,A,grnsample);

        %calculation of time normalized reward

        saclength1=sqrt((A(1)-x0)*(A(1)-x0)+(A(2)-y0)*(A(2)-y0));

        lprob= lprob+(log(DMC(A(2),A(1))+regular)-log(1/(dim*dim)));  
         %disp((log(DMC(A(2),A(1))+regular)-log(1/(dim*dim))));
     else 
        %starting new trial with prior
        eprob(end+1)=lprob;
        j=j+2;
        A=[Xtraj(j),Ytraj(j)];
        OX=0*OX;

    end            
    
    end
    %save gradients           
    Obmat(pen,repoch)=lprob/min(length(Xtraj),limitr);
    probseq{repoch}=eprob;
end     
prseq{end+1}=probseq;
% div=5;
% [ score ] = classperf(probseq,div);

end
figure;
imagesc(Obmat);
 set(gca,'YDir','normal');
 
amd=7;
cond=11;
accdiv=[];
truepos=[];
falsepos=[];
for div=1:500
disp(div);
scrs=cell(1,length(prseq));
parfor i=1:length(prseq)
    if div>length(prseq{i}{1})-1
        div0=length(prseq{i}{1})-1;
    else 
        div0=div;
    end
    scrs{i}=classperf(prseq{i},div0);
end

accdiv(end+1)=accuracy(scrs,amd,cond);
truepos(end+1)=trueposi(scrs,amd,cond);
falsepos(end+1)=falseposi(scrs,amd,cond);
end
figure
plot(accdiv);
figure
plot(truepos);
figure
plot(falsepos);