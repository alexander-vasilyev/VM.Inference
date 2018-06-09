addpath('MDP');
addpath('convolutions');
addpath('inference');
addpath('interface');
addpath('MonteCarlo');

%number of cells
dim=48;
dimm=2*dim-1;

circ=false;
stats=false;
%symmetric excecution function
alsym=true;
%dimensionality of Kernel
N=1;
AN=1;
%number of modalities
MCsteps=16*512;

M=1;
%display steps
gr=false;
%display maps
mgr=true;
%sim eye movements
seye=true;



devi= ones(2*N*(AN+1),M);

filt=1;
smdst=1;
gamma=0.00;

%number of epochs
epnum=1;
eplength=1;


%pertubation and learning coefficient
if gr
pert=0;
else 
pert=4;
end;
if seye
    step=5;
else
    step=4;
end;

%for parallel excecution
   if (gr==false) && eplength>1
    delete(gcp('nocreate'))
    parpool('local',eplength,'SpmdEnabled',false);
   end;
% reset(gpuDevice(1));
 

imbord=dim;
JSdist=repmat(zeros, [5 epnum]);
Nmdist=repmat(zeros, [5 epnum]);
Obdist=repmat(zeros, [5 epnum]);
Nadist=repmat(zeros, [5 epnum]);
Cndist=repmat(zeros, [5 epnum]);

%index extraction in convolutions
indmat=[];
for j=1:dim*dim
    jblock=floor((j-0.5)/dim);
    indmat(end+1)=j+jblock*(dim-1);
end
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
   


for pen=1:1

%average value
avvalue=zeros(20,1);
%average saccade number
avnumber=0;
%select sequence
Xtraj=trajcl{pen}(1,:);
Ytraj=trajcl{pen}(2,:);

%    RFun1= Radialupdate2(avpen(pen));
%RFun=Radialupdate2(0,dim,alsym);

%     Radi=avlengh(pen);
%     if Radi>23
%          Radi=23;
%     end;
 if seye
     Radi=6;
 else
    Radi=ceil(avlength(pen)/2);
 
 end
 Radi=12;
[CM,SM]=CircBasis(dim,N,AN,Radi); 
mixing=ones(2*N*(AN+1),M)/M;
% create basis of visibility map function
%if exist('G')==0

%end;

%if exist('randblock1')==0
%  rng('shuffle');
%  randblock1=randn(dim*dim,MCsteps*length(Xtraj)/blockdiv,'gpuArray');
%  rng('shuffle');
%  randblock2=randn(dim*dim,MCsteps*length(Xtraj)/blockdiv,'gpuArray');
%end
%initiate random starting visibility map
 rng('shuffle');
 hmeans=(randn(2*N*(AN+1),M))*60;
  hmeans(2:2*N*(AN+1))=0*hmeans(2:2*N*(AN+1));
  hmeans(1)=80;
 % hmeans(2)=80;
 % hmeans=meanc1{pen}; 
 if seye && mgr
 tarmeans=meanc1{pen};
 HFunt=IniHankel(tarmeans,CM,SM,dim,N,AN);    
 HFunt=abs(HFunt)+0.001;    
 str='Target visibility map';
 drawimage(HFunt,imbord,str,dim);
 end;
% hmeans=meanc1{pen};
%    hmeans(1:5*N)=0*hmeans(1:5*N);
%    hmeans(2)=30;
% hmeans((N+1):5*N,1:M)=0*hmeans((N+1):5*N,1:M);
% hmeans=meanc1{pen};
  hmean=hmeans;
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
    
    lpsum= zeros(1,eplength);
    vlpsum=zeros(1,eplength);
    devi1=expln(devi,2*N*(AN+1),M)/2;
     
    eventtot=zeros(dimm,dimm,eplength);
    eventfre=zeros(dimm,dimm,eplength);
    
    
    %initiate paramter spliting
    %random block generation
    %t = tic;
    parfor rep=1:eplength
        %sum of log-probability for sequence
        
       
        lprob=0;
        vlprob=0;
        %log probabilities for two parameter spliting
        lpone=0;
        lptwo=0;
        %initiate paramter spliting
               
        %hdifmean((N+1):5*N)=0*hdifmean((N+1):5*N);    
        [modal, hmean,dev]=selectmodal(mixing,hmeans,devi1,2*N*(AN+1));
        
        hdifmean= (dev'.*randn(1,2*N*(AN+1)))';
        hmeanone=hmean+hdifmean*pert;
        hmeantwo=hmean-hdifmean*pert;
        %create two visibility maps according to spliting
        D21=IniHankel(hmeanone,CM,SM,dim,N,AN);
        D22=IniHankel(hmeantwo,CM,SM,dim,N,AN);

        D21=abs(D21)+0.001;
        D22=abs(D22)+0.001;
        
        D2=D21;
        D1=D2.*D2;
        HFun1=rot90(rot90(D1));
        HFun1=HFun1(round(dim/2):(round(dim/2)+dim-1),round(dim/2):(round(dim/2)+dim-1));
        [Pfn,Hfn, fsize] = padarrays(P, HFun1,'same');
        visdft1=fft2(Hfn);
        visdfts1=fft2(Hfn.*Hfn);
        toep1=toepmat(HFun1,dim);
        
        D2=D22;
        D1=D2.*D2;
        HFun2=rot90(rot90(D1));
        HFun2=HFun2(round(dim/2):(round(dim/2)+dim-1),round(dim/2):(round(dim/2)+dim-1));
        [Pfn,Hfn, fsize] = padarrays(P, HFun2,'same');
        visdft2=fft2(Hfn);
        visdfts2=fft2(Hfn.*Hfn);
        toep2=toepmat(HFun2,dim);
        %evaluate log-probabilities for two maps
        for i= 1:2
            A=[Xtraj(2),Ytraj(2)];
            events=zeros(dimm,dimm);
            eventc=zeros(dimm,dimm);
            %OX=0*randn(dim*dim,MCsteps,'gpuArray');
            OX= gpuArray(repmat(zeros,[dim*dim,MCsteps]));
            if (i==1)
                visdft=visdft1;
                visdfts=visdfts1;
                HFun=HFun1;
                D2=gpuArray(D21);
                
                ftoep=gpuArray((toep1(lm:(lm+dim*dimm),:)));
                ftoeptr=toep1';                
                ftoeptr=gpuArray((ftoeptr(:,lm:(lm+dim*dimm))));                             
            else 
                visdft=visdft2;
                visdfts=visdfts2;
                HFun=HFun2;
                D2=gpuArray(D22); 
                
                ftoep=gpuArray((toep2(lm:(lm+dim*dimm),:)));
                ftoeptr=toep2';
                ftoeptr=gpuArray((ftoeptr(:,lm:(lm+dim*dimm))));                
            end

            j=2;
                                   
            while j<(length(Xtraj)-2)                        
            An=[Xtraj(j+1),Ytraj(j+1)]; 
            disp(j);
            if (An(1)*An(1)+An(2)*An(2))>0
                
                [ meanMC,diagMC,OX ] = inferenceMCGPUfull(A,D2,OX,MCsteps,dim); 
               
                % expected mean of information gain at each location

                %Imean=conv_fft2(meanMC',HFun,visdft,'same');
                
                %Imean=gpuArray(Imean);
                Imean=ftoep*meanMC;
                Imean=Imean(indmat);
                
                %[Einf, Dinf]=eigs(gather(diagMC),100);
                %sdiagMC=real(Einf*sqrt(Dinf)*Einf');             
                
                
                Infcov = toepcovmat( ftoep,ftoeptr,diagMC,indmat,dim);
                %Infcov = stdinf(ftoeptr, diagMC,dim,indmat);
                
                %Infcov=gpuArray(repmat(1, [2304 2304]));
                if gr
%                 mn=gather(meanMC); 
%                 mn=vec2mat(mn,dim);
%                 str='average probability';    
%                 drawimage(mn',imbord,str,dim);
%                 hold on;
%                 plot(A(1),A(2),'g*');
%                 pause(2);    
%                  
%                 
%                 str='variance of probability';    
%                 dmc=(gather(diagMC));
%                 dmc=diag(dmc);
%                 dmc=vec2mat(dmc,dim);
%                 drawimage(dmc',imbord,str,dim);
%                 hold on;
%                 plot(A(1),A(2),'g*');
%                 pause(2); 
                
                str='average information gain';
                imn=gather(Imean);
                imn=vec2mat(imn,dim);
                drawimage(imn',imbord,str,dim);
                hold on;
                plot(A(1),A(2),'g*');
                pause(2);
                
                str='variance of information gain';
                infc=gather(diag(Infcov));
                infc=vec2mat(infc,dim);
                drawimage(infc',imbord,str,dim);
                hold on;
                plot(A(1),A(2),'g*');
                pause(2);
                
                
%                 str='variance of information gain';
%                 drawimage(H,imbord,str,dim);
%                 hold on;
%                 plot(A(1),A(2),'g*');
%                 pause(2);
                end
               
                % variance of information gain at each location
               
                x0=A(1); 
                y0=A(2);
               
                %select next location
                A=[Xtraj(j+1),Ytraj(j+1)]; 
                j=j+1;
                
                
                %Infcov=sqrt(H(:));
                [Einf, Dinf]=eigs(gather(Infcov),30);
                Infcov=real(Einf*sqrt(Dinf)*Einf');
                DMC=decisionMCGPUfull(Imean,Infcov,MCsteps,gr,dim,filt,A);
                              
                %calculation of time normalized reward
                                                
                saclength1=sqrt((A(1)-x0)*(A(1)-x0)+(A(2)-y0)*(A(2)-y0));
                
                lprob= lprob+(log(DMC(A(2),A(1))+regular)-log(1/(dim*dim)))/length(Xtraj);  
                
                if (saclength1>1)
                    events(dim+A(2)-y0,dim+A(1)-x0)=events(dim+A(2)-y0,dim+A(1)-x0)+1;
                    eventc((dim+1-y0):(2*dim-y0),(dim+1-x0):(2*dim-x0))=eventc((dim+1-y0):(2*dim-y0),(dim+1-x0):(2*dim-x0))+DMC;                
%                        str='df';
%                        drawimage(eventc,imbord,str,dim);
%                        pause(1);
%                        disp(x0);
%                        disp(y0);
%                        str='df1';
%                        drawimage(events,imbord,str,dim);
%                        pause(1);
                end
            else 
                    %starting new trial with prior
                    
                j=j+2;
                A=[Xtraj(j),Ytraj(j)];
                OX=0*randn(dim*dim,MCsteps,'gpuArray');
                                   
            end            
            end
            
            if i==1
                lpone = lprob;                  
            else
                lptwo = lprob;                
            end
            lprob=0;  
            vlprob=0;
            %save events
            eventtot(:,:,rep)=eventtot(:,:,rep)+events;
            eventfre(:,:,rep)=eventfre(:,:,rep)+eventc;
        end 
        %save gradients
        defmeansum(:,rep)=hdifmean;
        linegradientsum(rep)= ((lpone-lptwo))/2;
        lpsum(rep)= ((lpone+lptwo))/2;
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

    meanlp= mean((lpsum));
    meanaver(end+1)=meanlp;
    baseline = meanlp;

    if repoch>M        
        baseline=sum(meanaver((repoch-M):(repoch-1)))/M;
    end
    %update of parameters
    %Hessian=Hessianmat(linegradientsum,lpsum,defmeansum,N,eplength);
    gradi=defmeansum*linegradientsum';
    %invHess=inv(Hessian'*Hessian +  0.000001*eye(size(Hessian)))*Hessian';
   
    %hupdate= -invHess*gradi;
    hupdate= step*gradi;
    %[mixing, hmeans, hupdate, devi] = parameterupdate(mixing,hmeans,5*N,M,vlpsum,mixgradientsum,linegradientsum,devi,baseline,step,defmeansum,eplength);
    hmeans=hmeans-gamma*hmeans+hupdate;
    
    eventfrec=eventfrec/sum(sum(eventfrec));
    eventtotc=eventtotc/sum(sum(eventtotc));
    eventfrec=imgaussfilt(eventfrec,smdst);
    %show target distribution
   % if (repoch<2 && mgr)
    str='Target distribution'; 
     drawimage(eventfrec,imbord,str,dim); 
    tardist{end+1}=eventfrec;
   % end
    %show final distribution
   % if (repoch==epnum && mgr)
    str='Simulated distribution';
    drawimage(eventtotc,imbord,str,dim);
    infdist{end+1}=eventtotc;
   % end
    %jenssen shannon between target and simulated distributions
    diffev=kld(eventfrec,eventtotc);
    %save coeffients
    hmean=sum(hmeans.*mixing,2);
    
    HFun=IniHankel(hmean,CM,SM,dim,N,AN);    
    HFun=abs(HFun)+0.01; 
   
    if (repoch==epnum && mgr)    
    str='Inferred visibility map';
    drawimage(HFun,imbord,str,dim);
    infval{end+1}=HFun;
    end
%evaluate distance between maps
%     if seye
%     [ dhf, dhf1 ] = distanceVM( HFun,HFunt,Radi);
%     end
    
    if repoch==epnum
    [hst, crr]=angcorr(HFun,eventtotc,eventfrec,dim);
    corrval{end+1}=crr;
    histval{end+1}=hst;
    end
    %mean completion time, square of VM, jensson shannon distance to
    %target, and distance between target VM and current
    
   % concen=concentration(mixing,M,5*N);
    if seye
    Nmdist(pen,repoch)=dhf;
    Nadist(pen,repoch)=dhf1;
    end
    JSdist(pen,repoch)=diffev;
    Obdist(pen,repoch)=baseline;
   % Cndist(pen,repoch)=concen;
    if seye
    output= [meanlp, baseline, diffev, dhf, dhf1,sum(hupdate.*hupdate)];
    else
    output= [meanlp, baseline, diffev,sum(hupdate.*hupdate)];
    end  
    
    disp(output);
end                      
end
 
if stats
figure
jsdistance=sum(JSdist,1)/length(trajcl);
plot(jsdistance);
title('Average JS distance');
drawnow;
  

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


figure
obdistance=sum(Obdist,1)/length(trajcl);
plot(obdistance);
title('Average objective change');
drawnow;        


figure
cndistance=sum(Cndist,1)/length(trajcl);
plot(cndistance);
title('Average concentration');
drawnow;  
end;