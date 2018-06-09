addpath('MDP');
addpath('convolutions');
addpath('filter');
addpath('inference');
addpath('interface');
addpath('MonteCarlo');

%number of cells
dim=32;
dimm=2*dim-1;
circ=true;
stats=true;
alsym=true;
%dimensionality of Kernel
N=2;
AN=1;
%number of modalities
MCsteps=128;

M=1;
%display steps
gr=false;
%display maps
mgr=true;
%sim eye movements
seye=true;

%size of validation set
valid=0.0;

devi= ones(2*N*(AN+1),M);

filt=3;
smdst=1;
gamma=0.00;

%number of epochs
epnum=10;
eplength=4;


%pertubation and learning coefficient
if gr
pert=0;
else 
pert=5;
end;

if seye
    step=5;
else
    step=5;
end;

%for parallel excecution
     if (gr==false) && eplength>1
       delete(gcp('nocreate'));
          reset(gpuDevice(2));
          g=gpuDevice(2);
       parpool('local',eplength,'SpmdEnabled',false);
      end;
% reset(gpuDevice(1));
 

imbord=dim;
CHdist=repmat(zeros, [5 epnum]);
Pvdist=repmat(zeros, [5 epnum]);
Jsdist=repmat(zeros, [5 epnum]);
Nmdist=repmat(zeros, [5 epnum]);
Obdist=repmat(zeros, [5 epnum]);
Nadist=repmat(zeros, [5 epnum]);
Cndist=repmat(zeros, [5 epnum]);
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

infval={};
infdist={};
tardist={};
corrval={};
histval={};

P=rand(dim);
%initial probability
if circ
    P0 = InitialP(dim);
else 
    P0 = repmat(1/(dim*dim) , [dim dim]);
end
  

%jd=sum(jdist,2);
%jdist=jdist./jd;

for pen=1:length(trajcl)

%select sequence
Xtraj=trajcl{pen}(1,:);
Ytraj=trajcl{pen}(2,:);
% 
%  if seye
%     Radi=8;
%  else
%     Radi=ceil(avlength(pen)/2); 
%  end
 Radi=5;
[CM,SM]=CircBasis(dim,N,AN,Radi); 
Radi2=32;
[CM2,SM2]=CircBasis(dim,N,AN,Radi2); 

%initiate random starting visibility map
 rng('shuffle');
 hmeans=(randn(2*N*(AN+1),M))*40;
  hmeans(2:2*N*(AN+1))=0*hmeans(2:2*N*(AN+1));
  hmeans(1)=10/sqrt(pen);
 MCsteps=MCsteps/pen;
 %hmeans=meanc1{pen}*2; 
 if seye && mgr   
 HFunt=Hinival{pen};    
 str='Target visibility map';
 drawimage(HFunt,imbord,str,dim);
 end;

  hmean=hmeans;
  
  rmeans=0*(randn(2*N*(AN+1),M))*40;
  rmeans(1)=0;
  %hmean=sum(hmeans.*mixing,2);
    %evaluate visibility map
  HFun=IniHankel(hmeans,CM,SM,dim,N,AN);     
  HFun=abs(HFun)+0.001; 
    %show final visibility map
  if mgr
  str='Initial visibility map';
  drawimage(HFun,imbord,str,dim);
  end;
  if seye
  [ dhf, dhf1 ] = distanceVM( HFun,HFunt,Radi);
  disp(dhf);
  disp(dhf1);
  end;

%distribution functions
eventtotc=zeros(dimm,dimm);
eventfrec=zeros(dimm,dimm);
rng(1,'simdTwister');

%save coefficient for different epochs

meanaver=[];
%start evaluation for selected sequence of eye-movements 

for repoch = 1:epnum


    %gradient function
    linegradientsum= zeros(1,eplength);
    mixgradientsum=zeros(2*N*(AN+1),eplength);	
    defmeansum=zeros(2*N*(AN+1),eplength);
    rdefmeansum=zeros(2*N*(AN+1),eplength);
    
    lpsum= zeros(1,eplength);
    vlpsum=zeros(1,eplength);
    devi1=expln(devi,2*N*(AN+1),M)/2;
     
    eventtot=zeros(dimm,dimm,eplength);
    eventfre=zeros(dimm,dimm,eplength);
    
   
    parfor rep=1:eplength
        %sum of log-probability for sequence
      
       
        lprob=0;
        vlprob=0;
        %log probabilities for two parameter spliting
        lpone=0;
        lptwo=0;
        vlpone=0;
        vlptwo=0;
        %initiate paramter spliting
         probarray=[];      
        %hdifmean((N+1):5*N)=0*hdifmean((N+1):5*N);    
        [modal, hmean,dev]=selectmodal(mixing,hmeans,devi1,2*N*(AN+1));
        
        hdifmean= (dev'.*randn(1,2*N*(AN+1)))';
        hmeanone=hmean+hdifmean*pert;
        hmeantwo=hmean-hdifmean*pert;
        rdifmean= (dev'.*randn(1,2*N*(AN+1)))';
        rmeanone=rmeans+rdifmean*pert;
        rmeantwo=rmeans-rdifmean*pert;
        
        %create two visibility maps according to spliting
        D21=IniHankel(hmeanone,CM,SM,dim,N,AN);
        D22=IniHankel(hmeantwo,CM,SM,dim,N,AN);

        D21=abs(D21)+0.001;
        D22=abs(D22)+0.001;
        
        RFun1=abs(IniHankel2(rmeanone,CM2,SM2,dim,N,AN)+10);
        RFun2=abs(IniHankel2(rmeantwo,CM2,SM2,dim,N,AN)+10);
        
        D2=D21;
        D1=D2.*D2;
        HFun1=rot90(rot90(D1));
        HFun1=HFun1(round(dim/2):(round(dim/2)+dim-1),round(dim/2):(round(dim/2)+dim-1));
        [Pfn,Hfn, fsize] = padarrays(P, HFun1,'same');
        visdft1=fft2(Hfn);
        visdfts1=fft2(Hfn.*Hfn);
        
        
        D2=D22;
        D1=D2.*D2;
        HFun2=rot90(rot90(D1));
        HFun2=HFun2(round(dim/2):(round(dim/2)+dim-1),round(dim/2):(round(dim/2)+dim-1));
        [Pfn,Hfn, fsize] = padarrays(P, HFun2,'same');
        visdft2=fft2(Hfn);
        visdfts2=fft2(Hfn.*Hfn);
        
        %evaluate log-probabilities for two maps
        for i= 1:2
            A=[Xtraj(2),Ytraj(2)];
            events=zeros(dimm,dimm);
            eventc=zeros(dimm,dimm);
            
            OX= gpuArray(repmat(zeros,[dim*dim,MCsteps]));
            if (i==1)
                visdft=visdft1;
                visdfts=visdfts1;
                HFun=HFun1;
                D2=gpuArray(D21);
                RFun=RFun1;
            else 
                visdft=visdft2;
                visdfts=visdfts2;
                HFun=HFun2;
                D2=gpuArray(D22); 
                RFun=RFun2;
            end

            j=2;
                                 
            while (j<(length(Xtraj)-2))                       
            An=[Xtraj(j+1),Ytraj(j+1)]; 
            
            if (An(1)*An(1)+An(2)*An(2))>0
                
                grnsample=randn(dim*dim,MCsteps,'gpuArray');
                
                [ meanMC,diagMC,OX ] = inferenceMCGPUdiag(A,D2,OX,dim,grnsample,MCsteps); 
               
                % expected mean of information gain at each location

                Imean=conv_fft2(meanMC',HFun,visdft,'same',dim,fsize);
                                           
                H= conv_fft2(diagMC',HFun,visdfts,'same',dim,fsize);
                 
                Imean = reward(Imean,A,RFun,dim);
                
                H=reward(H,A,RFun.*RFun,dim);
                if gr

                str='average information gain';
                drawimage(Imean,imbord,str,dim);
                hold on;
                plot(A(1),A(2),'g*');
                pause(2);
                                  
                str='variance of information gain';
                drawimage(H,imbord,str,dim);
                hold on;
                plot(A(1),A(2),'g*');
                pause(2);
                end
               
                % variance of information gain at each location
               
                x0=A(1); 
                y0=A(2);
               
                %select next location
                A=[Xtraj(j+1),Ytraj(j+1)]; 
                j=j+1;
                
                Imean=gpuArray(Imean(:));
                Infcov=gpuArray(sqrt(H(:)));

                DMC=decisionMCGPUdiag(Imean,Infcov,gr,dim,filt,A,grnsample,MCsteps);
                              
                %calculation of time normalized reward
                                                
                saclength1=sqrt((A(1)-x0)*(A(1)-x0)+(A(2)-y0)*(A(2)-y0));
                
                if j<ceil( (1-valid)*length(Xtraj))
                    lprob= lprob+(log(DMC(A(2),A(1))+regular)-log(1/(dim*dim)))/length(Xtraj);  
                else
                    vlprob=vlprob+(log(DMC(A(2),A(1))+regular)-log(1/(dim*dim)))/length(Xtraj);  
                end
                probarray(end+1)=lprob;
                %disp((log(DMC(A(2),A(1))+regular)-log(1/(dim*dim))));
               % if (saclength1>1)
                    events(dim+A(2)-y0,dim+A(1)-x0)=events(dim+A(2)-y0,dim+A(1)-x0)+1;
                    eventc((dim+1-y0):(2*dim-y0),(dim+1-x0):(2*dim-x0))=eventc((dim+1-y0):(2*dim-y0),(dim+1-x0):(2*dim-x0))+DMC;                
               % end
            else 
                %starting new trial with prior
                    
                j=j+2;
                A=[Xtraj(j),Ytraj(j)];
                OX=0*OX;
                                   
            end            
            end
            
            if i==1
                lpone = lprob;       
                vlpone=vlprob;
            else
                lptwo = lprob;  
                vlptwo=vlprob;
            end
            lprob=0;  
            vlprob=0;
            %save events
            eventtot(:,:,rep)=eventtot(:,:,rep)+events;
            eventfre(:,:,rep)=eventfre(:,:,rep)+eventc;
        end 
        %save gradients
        defmeansum(:,rep)=hdifmean;
        rdefmeansum(:,rep)=rdifmean;
        linegradientsum(rep)= (lpone-lptwo)/2;
        lpsum(rep)= (lpone+lptwo)/2;
        vlpsum(rep)=(vlpone+vlptwo)/2;
        mixgradientsum(:,rep)=modal;
        
    end
    %ITERATION_TIME = toc(t)
    eventfrec=ones(dimm,dimm);
    eventtotc=ones(dimm,dimm);
    for i=1:eplength
        eventfrec=eventfrec+eventtot(:,:,i);
        eventtotc=eventtotc+eventfre(:,:,i);
    end


    %summation of gradients  

    meanlp= mean(lpsum);
    meanvlp=mean(vlpsum);
  
    gradi=defmeansum*linegradientsum'/(eplength/4);
    
    hupdate= step*gradi;
   
    hmeans=hmeans-gamma*hmeans+hupdate;
    
    rgradi=rdefmeansum*linegradientsum'/(eplength/4);
    
    rupdate= step*rgradi;    
    rmeans=rmeans+ rupdate;
    
    eventfrec=eventfrec/sum(sum(eventfrec));
    eventtotc=eventtotc/sum(sum(eventtotc));
    eventfrec=imgaussfilt(eventfrec,smdst);
    %show target distribution
    if (repoch<2 && mgr)
    str='Target distribution'; 
     drawimage(eventfrec,imbord,str,dim); 
    tardist{end+1}=eventfrec;
    end
    %show final distribution
    if (repoch==epnum && mgr)
    str='Simulated distribution';
    drawimage(eventtotc,imbord,str,dim);
    infdist{end+1}=eventtotc;
    end
    %jenssen shannon between target and simulated distributions
    jsdif=kld(eventfrec,eventtotc);
    diffev=chisq(eventfrec,eventtotc);
    pval=1-chi2cdf(diffev,2);
    %save coeffients
    hmean=sum(hmeans.*mixing,2);
    
    HFun=IniHankel(hmean,CM,SM,dim,N,AN);    
    HFun=abs(HFun)+0.01; 
   
    %if (repoch==epnum && mgr)    
    str='Inferred visibility map';
    drawimage(HFun,imbord,str,dim);
    infval{end+1}=HFun;
    
    RFun=abs(IniHankel2(rmeans,CM2,SM2,dim,N,AN)+10);
    str='Inferred radial function';
    drawimage(RFun,imbord,str,dim);
    %end
%evaluate distance between maps
     if seye
     [ dhf, dhf1 ] = distanceVM( HFun,HFunt,Radi);
     end
    
    if repoch==epnum
  %  [hst, crr]=angcorr(HFun,eventtotc,eventfrec,jdist(pen,:),dim,4);
  %  corrval{end+1}=crr;
  %  histval{end+1}=hst;
    end
    %mean completion time, square of VM, jensson shannon distance to
    %target, and distance between target VM and current
    
   % concen=concentration(mixing,M,5*N);
    if seye
    Nmdist(pen,repoch)=dhf;
    Nadist(pen,repoch)=dhf1;    
    end
    Jsdist(pen,repoch)=jsdif;
    CHdist(pen,repoch)=diffev;
    Pvdist(pen,repoch)=pval;
    Obdist(pen,repoch)=meanlp;
    vObdist(pen,repoch)=meanvlp;
   % Cndist(pen,repoch)=concen;
    if seye
    output= [meanlp, meanvlp, diffev,pval, dhf, dhf1,sum(hupdate.*hupdate)];
    else
    output= [meanlp, meanvlp, diffev,pval, sum(hupdate.*hupdate)];
    end  
    
    disp(output);
end                      
end
 
if stats
figure
chdistance=sum(CHdist,1)/length(trajcl);
plot(chdistance);
title('Average Chi-square');
drawnow;
  

figure
pvdistance=sum(Pvdist,1)/length(trajcl);
plot(pvdistance);
title('Average p-value');
drawnow;

figure
jsdistance=sum(Jsdist,1)/length(trajcl);
plot(jsdistance);
title('Average J-S distance');
drawnow;

if seye
    figure
    nmdistance=sum(Nmdist,1)/length(trajcl);
    plot(nmdistance);
    title('Average distance between normalized visibility maps');
    drawnow;       


    figure
    nadistance=sum(Nadist,1)/length(trajcl);
    plot(nadistance);
    title('Average distance between visibility maps');
    drawnow;  
end;

figure
obdistance=sum(Obdist,1)/length(trajcl);
plot(obdistance);
title('Average objective change');
drawnow;        

figure
vobdistance=sum(vObdist,1)/length(trajcl);
plot(vobdistance);
title('Average objective change on validation set');
drawnow;        
end;