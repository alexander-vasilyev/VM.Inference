addpath('MDP');
addpath('convolutions');
addpath('inference');
addpath('interface');
addpath('MonteCarlo');

%number of cells
dim=32;
dimm=2*dim-1;

circ=false;

%symmetric excecution function
alsym=true;
%dimensionality of Kernel
N=1;
AN=1;
%number of modalities
MCsteps=16*512;

M=1;
%display steps
gr=true;
%display maps
mgr=true;
%sim eye movements
seye=true;



devi= ones(2*N*(AN+1),M);

filt=0;
smdst=1;
gamma=0.00;

%number of epochs
epnum=17;
eplength=1;


%pertubation and learning coefficient
if gr
pert=0;
else 
pert=0.5;
end;
if seye
    step=0.5;
else
    step=0.5;
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
end;
   

for pen=1:length(trajcl)

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
     Radi=10;
 else
    Radi=avlength(pen);
 end
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
 hmeans=(randn(2*N*(AN+1),M))*20;
% hmeans(2:2*N*(AN+1))=0*hmeans(2:2*N*(AN+1));
% hmeans(1)=10;
 %hmeans=meanc1{pen}; 
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

  hmean=sum(hmeans.*mixing,2);
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
                ftoeptr=toep1';                
                ftoeptr=gpuArray((ftoeptr(:,lm:(lm+dim*dimm))));                             
            else 
                visdft=visdft2;
                visdfts=visdfts2;
                HFun=HFun2;
                D2=gpuArray(D22);               
                ftoeptr=toep2';
                ftoeptr=gpuArray((ftoeptr(:,lm:(lm+dim*dimm))));                
            end

            j=2;
            
            
            while j<(length(Xtraj)-2)
            
            An=[Xtraj(j+1),Ytraj(j+1)]; 
            if (An(1)*An(1)+An(2)*An(2))>0
                % concatenation function for visibility (optimisation reasons)
                DC= conc(D2,A(1),A(2),dim);           
                % estimation of the observation vector - response from each location
                jk=round(rand*rnseed);
                jk0=round(rand*MCsteps);
                % inference corresponding to observation 
                rnblock1=randblock1(:,(1+jk*MCsteps+jk0):((jk+1)*MCsteps+jk0));
                [ meanMC,diagMC,OX ] = inferenceMC(DC,OX,MCsteps,gr,dim,rnblock1); 

                
                % expected mean of information gain at each location
                
                Imean= conv_fft2(meanMC,HFun,visdft,'same');
                
                if gr
                str='average information gain';
                drawimage(Imean,imbord,str,dim);
                hold on;
                plot(A(2),A(1),'g*');
                pause(2);
                end;
                
                %Imean1=toepcnv(ftoep,meanMC,dim);
                

                %Infcov=toepcovmat( ftoep,ftoeptr,diagMC,dim,indmat);
                [ Infcov ] = stdinf(ftoeptr,diagMC,dim,indmat);
                
                % variance of information gain at each location
                
                %[Einf, Dinf]=eig(Infcov);
                %Infcov=real(Einf*sqrt(Dinf)*Einf');
                if gr
                Icov= conv_fft2(diagMC,HFun,visdfts,'same'); 
                
                str='variance of information gain';
                drawimage(Icov,imbord,str,dim);
                hold on;
                plot(A(2),A(1),'g*');
                pause(2);
                
                
                ghj=diag(Infcov);
                ghj=vec2mat(ghj,dim);
                drawimage(flipud(ghj),imbord,str,dim);
                end;
                x0=A(1); 
                y0=A(2);
                A0=A;
                %select next location
                A=[Xtraj(j+1),Ytraj(j+1)]; 
                j=j+1;
                
                jl=round(rand*rnseed);
                jl0=round(rand*MCsteps);
                
                rnblock2=randblock2(:,(1+jl*MCsteps+jl0):((jl+1)*MCsteps+jl0));
                DMC=decisionMC(Imean,Infcov,MCsteps,gr,dim,rnblock2);
                [ADx,ADy] = find(DMC==max(DMC(:)));
                %calculation of time normalized reward
                if gr
                hold on;
                plot(A(2),A(1),'b*');
                pause(5);
                end;

                saclength1=sqrt((A(1)-x0)*(A(1)-x0)+(A(2)-y0)*(A(2)-y0));
                %disp(log(DMC(A(1),A(2))+regular)-log(1/(dim*dim)));
                lprob= lprob  + (log(DMC(A(1),A(2))+regular)-log(1/(dim*dim)))/length(Xtraj);  
                

                %if (saclength1>1)
                    events(dim+A(1)-x0,dim+A(2)-y0)=events(dim+A(1)-x0,dim+A(2)-y0)+1;%exp(-((x-A(1))*(x-A(1))+(y-A(2))*(y-A(2)))/(8+saclength+saclength*saclength/16));
                    eventc((dim+1-x0):(2*dim-x0),(dim+1-y0):(2*dim-y0))=eventc((dim+1-x0):(2*dim-x0),(dim+1-y0):(2*dim-y0))+DMC;%exp(-((x-A(1))*(x-A(1))+(y-A(2))*(y-A(2)))/(8+saclength+saclength*saclength/16));                   
                %end
            else 
                    %starting new trial with prior
                    
                j=j+2;
                A=[Xtraj(j),Ytraj(j)];
                OX=repmat(zeros,[dim*dim,MCsteps]);
                    
            end;


            end;
            
            if i==1
                lpone= lprob;  
                
            else
                lptwo = lprob; 
                
            end;
            lprob=0;  
            vlprob=0;
            %save events
            eventtot(:,:,rep)=eventtot(:,:,rep)+events;
            eventfre(:,:,rep)=eventfre(:,:,rep)+eventc;
        end;  
        %save gradients
        defmeansum(:,rep)=hdifmean;
        mixgradientsum(:,rep)=modal;
        linegradientsum(rep)= ((lpone-lptwo))/10;
        lpsum(rep)= ((lpone+lptwo))/10;

    end;
    eventfrec=ones(dimm,dimm);
    eventtotc=ones(dimm,dimm);
    for i=1:eplength
        eventfrec=eventfrec+eventtot(:,:,i);
        eventtotc=eventtotc+eventfre(:,:,i);
    end;

    ITERATION_TIME = toc(t)
    %summation of gradients  

    meanlp= mean((lpsum))/2;
    meanaver(end+1)=meanlp;
    baseline = meanlp;

    if repoch>M        
        baseline=sum(meanaver((repoch-M):(repoch-1)))/M;
    end;
    %update of parameters
     hupdate= defmeansum*linegradientsum'*step;
    hmeans=hmeans+hupdate-gamma*hmeans;
    %hmeans(1)=hmeans(1)+5;
    eventfrec=eventfrec/sum(sum(eventfrec));
    eventtotc=eventtotc/sum(sum(eventtotc));
    %show target distribution
    if repoch<2
    str='Target distribution';
    drawimage(eventfrec,imbord,str,dim);    
    end;
    %show final distribution
    if repoch==epnum
    str='Simulated distribution';
    drawimage(eventtotc,imbord,str,dim);
    end;
    %jenssen shannon between target and simulated distributions
    diffev=kld(eventfrec,eventtotc);
    %save coeffients
    hmean=sum(hmeans.*mixing,2);
    
    HFun=IniHankel(G,F,K ,hmean,CM,SM,C2M,S2M,N,dim);     
    HFun=(HFun.*HFun)+0.01; 
    
    
    
    if repoch==epnum    
    str='Inferred visibility map';
    drawimage(HFun,imbord,str,dim);
    end;
%evaluate distance between maps
    if seye
    [ dhf, dhf1 ] = distanceVM( HFun,HFunt,Radi);
    end;
    %mean completion time, square of VM, jensson shannon distance to
    %target, and distance between target VM and current
    
    concen=concentration(mixing,M,5*N);
    if seye
    Nmdist(pen,repoch)=dhf;
    Nadist(pen,repoch)=dhf1;
    end;
    JSdist(pen,repoch)=diffev;
    Obdist(pen,repoch)=baseline;
    Cndist(pen,repoch)=concen;
    if seye
    output= [meanlp, baseline, concen, diffev, dhf, dhf1,sum(hupdate.*hupdate)];
    else
     output= [meanlp, baseline, concen, diffev,sum(hupdate.*hupdate)];
    end;  
    
    disp(output);
end;
                       
end;
    
figure
jsdistance=sum(JSdist,1)/20;
plot(jsdistance);
title('Average JS distance');
drawnow;
  

figure
nmdistance=sum(Nmdist,1)/20;
plot(nmdistance);
title('Average distance between normalized visibility maps');
drawnow;       


figure
nadistance=sum(Nadist,1)/20;
plot(nadistance);
title('Average distance between visibility maps');
drawnow;  


figure
obdistance=sum(Obdist,1)/20;
plot(obdistance);
title('Average objective change');
drawnow;        


figure
cndistance=sum(Cndist,1)/20;
plot(cndistance);
title('Average concentration');
drawnow;  
