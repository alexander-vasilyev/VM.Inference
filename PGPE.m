addpath('MDP');
addpath('convolutions');
addpath('filter');
addpath('inference');
addpath('interface');
addpath('MonteCarlo');

%number of cells
dim=32;
dimm=2*dim-1;
circ=false;
stats=true;
alsym=true;
%dimensionality of Kernel
N=2;
AN=1;
%number of modalities
MCsteps=2*512;

M=1;
%display steps
gr=false;
%display maps
mgr=true;
%sim eye movements


%size of validation set
valid=0.0;

devi= ones(2*N*(AN+1),M);

filt=4;
smdst=1;
gamma=0.00;

%number of epochs
epnum=20;
eplength=6;


%pertubation and learning coefficient
if gr
pert=0;
else 
pert=8;
end;


step=12;


%for parallel excecution
if (gr==false) && eplength>1
delete(gcp('nocreate'));
reset(gpuDevice(1));
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
  
jdist=load('visdistr.txt');
%jd=sum(jdist,2);
%jdist=jdist./jd;
Radi=5;
[CM,SM]=CircBasis(dim,N,AN,Radi); 
Radi2=32;
[CM2,SM2]=CircBasis(dim,N,AN,Radi2); 

RFunval={};
HFunval={};
Hinival={};
rmeans=0*(randn(2*N*(AN+1),M))*40;
rmeans(1)=0;
for pen=1:12

mixing=ones(2*N*(AN+1),M)/M;

%initiate random starting visibility map
rng('shuffle');
hmeans=(randn(2*N*(AN+1),M))*40;
hmeans(2:2*N*(AN+1))=0*hmeans(2:2*N*(AN+1));
hmeans(1)=50/sqrt(pen);


tarmeans=(randn(2*N*(AN+1),M))*40;
tarmeans(2:2*N*(AN+1))=0*tarmeans(2:2*N*(AN+1));
tarmeans(1)=50/sqrt(pen);

HFunt=IniHankel(tarmeans,CM,SM,dim,N,AN);    
HFunt=abs(HFunt)+0.001;    
% str='Visibility map';
% drawimage(HFunt,imbord,str,dim);
Hinival{end+1}=HFunt;

% rmeans=0*(randn(2*N*(AN+1),M))*40;
% rmeans(1)=0;
hmean=hmeans;
%hmean=sum(hmeans.*mixing,2);
%evaluate visibility map
HFun=IniHankel(hmeans,CM,SM,dim,N,AN);     
HFun=abs(HFun)+0.001; 
RFun=abs(IniHankel2(rmeans,CM2,SM2,dim,N,AN)+10);
%show final visibility map
if mgr
% str='Initial visibility map';
% drawimage(HFun,imbord,str,dim);
% str='Initial radial function';
% drawimage(RFun,imbord,str,dim);
end;


%distribution functions
eventtotc=zeros(dimm,dimm);
eventfrec=zeros(dimm,dimm);
rng(1,'simdTwister');

%save coefficient for different epochs

meanaver=[];
%start evaluation for selected sequence of eye-movements 
epnum=20-pen;
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
       
        P=P0;
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
        D2=HFunt;
        %evaluate log-probabilities for two maps
        for i= 1:2
           
            events=zeros(dimm,dimm);
            eventc=zeros(dimm,dimm);            
           
            if (i==1)
                visdft=visdft1;
                visdfts=visdfts1;
                HFun=HFun1;               
                RFun=RFun1;
            else 
                visdft=visdft2;
                visdfts=visdfts2;
                HFun=HFun2;                 
                RFun=RFun2;
            end

           
        % the initial probability - uniform
            for ij=1:500
            S= targetloc(dim,circ);
            A= targetloc(dim,circ);   
            P=P0;  
           
                while ((P(S(2),S(1))<0.5)&& (abs(lprob*100)<2000))
            %while ( time<1000)

                % concatenation function for visibility (optimisation reasons)
                DC= conc(D2,A(1),A(2),dim);           
                % estimation of the observation vector - response from each location
                [O, prob] = observation(S,A,DC,dim,true);   

                % inference corresponding to observation 
                P = inference(P,O,A,DC);
                if gr
                  str='Probability';
                  drawimage(P,imbord,str,dim);
                  pause(2);
                end
                % expected entropy estimation for each location
                H= conv_fft2(P,HFun,visdft,'same',dim,fsize); 

                %calculation of time normalized reward
                %R= H;
                R = reward(H,A,RFun,dim);
                
                
                
                if gr
                 str='Information gain';
                  drawimage(H,imbord,str,dim);
                  pause(2);
                 str='Reward function';
                  drawimage(R,imbord,str,dim);
                  pause(2);
                end
                % prior coordinates
                x0 = A(1);
                y0 = A(2);

                %estimation of best location to transfer the gaze
                [y,x] = find(R==max(R(:)));

    %             FG= softmax(R,dim);
    %             x=FG(1);
    %             y=FG(2);

                if (length(x)>1)
    %                   x(1)=S(1);                    
                    x(2:length(x))=[];                  
                end;    
                if (length(y)>1)
    %                  y(1)=S(2);
                    y(2:length(y))=[];
                end;
                
                %stochastic saccade placement, intended saccade length:
                intsaclength= sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));                  
               % A=stoch(A,intsaclength,dim);
                A(1)=x;
                A(2)=y;

                %real saccade length
                saclength= sqrt((A(1)-x0)*(A(1)-x0)+(A(2)-y0)*(A(2)-y0)); 
                %lhist(end+1)=saclength;
               % if saclength>1
                %eventtotc(dim+A(2)-y0,dim+A(1)-x0)=eventtotc(dim+A(2)-y0,dim+A(1)-x0)+1;
                if saclength>0
                    lprob = lprob + (125 + 25/pen + 0*saclength)/1000000;
                else
                    lprob= lprob + (25/1000000)/pen;
                end

                end
            end
            lprob=-lprob;
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

%     if (repoch<2 && mgr)
%     str='Target distribution'; 
%      drawimage(eventfrec,imbord,str,dim); 
%     tardist{end+1}=eventfrec;
%     end
    %show final distribution
%     if (repoch==epnum && mgr)
%     str='Simulated distribution';
%     drawimage(eventtotc,imbord,str,dim);
%     infdist{end+1}=eventtotc;
%     end
    %jenssen shannon between target and simulated distributions

    %save coeffients
    HFun=IniHankel(hmeans,CM,SM,dim,N,AN);     
    HFun=abs(HFun)+0.001;
   
    if (repoch==epnum && mgr)    
    str='Smoothing kernel';
    drawimage(HFun,imbord,str,dim);
    HFunval{end+1}=HFun;
    end

    RFun=abs(IniHankel2(rmeans,CM2,SM2,dim,N,AN)+10);
    
    if (repoch==epnum && mgr)    
    str='Radial function';
    drawimage(RFun,imbord,str,dim);
    RFunval{end+1}=RFun;
    end
   % concen=concentration(mixing,M,5*N);

    nHFun=HFun/sum(sum(HFun));
    nHFunt=HFunt/sum(sum(HFunt));
    CHdist(pen,repoch)=sum(sum((abs(nHFun-nHFunt))));
   
    Obdist(pen,repoch)=meanlp;
  
   % Cndist(pen,repoch)=concen;
    
    output= [meanlp,sum(sum((abs(nHFun-nHFunt)))), sum(hupdate.*hupdate)];
      
    
    disp(output);
end                      
end
figure
plot(sum(CHdist,1));
figure
plot(sum(Obdist,1));

        



