addpath('MDP');
addpath('convolutions');
addpath('inference');
addpath('interface');
addpath('MonteCarlo');


dim=96;
dimm=2*dim-1;
events=zeros(length(trajcl),dimm,dimm);
eventt=zeros(length(trajcl),dimm,dimm);
eventots=zeros(dimm,dimm);
eventott=zeros(dimm,dimm);
for pen=1:length(trajcl)

%select sequence
Xtraj=trajcl{pen}(1,:);
Ytraj=trajcl{pen}(2,:);
%timtrajc=timetrajcl{pen};
j=2;
A=[Xtraj(2),Ytraj(2)];
while j<(length(Xtraj)-2)              
            An=[Xtraj(j+1),Ytraj(j+1)]; 
            if (An(1)*An(1)+An(2)*An(2))>0                
                x0=A(1); 
                y0=A(2);               
                %select next location
                A=[Xtraj(j+1),Ytraj(j+1)]; 
                j=j+1;
                saclength1=sqrt((A(1)-x0)*(A(1)-x0)+(A(2)-y0)*(A(2)-y0));                
                if saclength1>0
                events(pen,dim+A(2)-y0,dim+A(1)-x0)=events(pen,dim+A(2)-y0,dim+A(1)-x0)+1;
               % eventt(pen,dim+A(2)-y0,dim+A(1)-x0)=eventt(pen,dim+A(2)-y0,dim+A(1)-x0)+timtrajc(j);
                eventots(dim+A(2)-y0,dim+A(1)-x0)=eventots(dim+A(2)-y0,dim+A(1)-x0)+1;
               % eventott(dim+A(2)-y0,dim+A(1)-x0)=eventott(dim+A(2)-y0,dim+A(1)-x0)+timtrajc(j);
                end;
            else 
                    %starting new trial with prior                    
                j=j+2;
                A=[Xtraj(j),Ytraj(j)];               
            end;
           
end;
% 
% figure
% imagesc(squeeze(events(pen,:,:))); 
% drawnow;
% figure
% imagesc(squeeze(eventt(pen,:,:))./squeeze(events(pen,:,:))); 
% drawnow;
end;
% figure
% speedmap=eventott(:,:)./(eventots(:,:)+0.001);
% imagesc(speedmap); 
% drawnow;

N=4;
AN=2;
respotions=repmat(0,[6,length(trajcl)]);
mnposition=repmat(0,[6,length(trajcl),2*N*(AN+1)]);
meanc1={};
for pen=1:length(trajcl)
    for j=1:8
        Radi=3*j;
        [ CM,SM ] = CircBasis( dim ,N,AN,Radi);
        hj={CM,SM};
        P=squeeze(events(pen,:,:));
        P=P/sum(sum(P));
        P=rot90(rot90(P));
        f=@computeSquare;   
        xd=(randn(2*N*(AN+1),1));
        [xd,resnorm,residual,exitflag,output]=lsqcurvefit(f,xd,hj,P);
        if j==1
            str='target distribution';
            drawimage(P,imbord,str,dim);
            
        end;
        
        respotions(j,pen)=resnorm;
        mnposition(j,pen,:)=200*xd;
    end;
end;    
[~,ii]=min(respotions);
avlength=repmat(1,1,length(trajcl));
for j=1:length(trajcl)
    avlength(j)=3*ii(j);
    meanc1{end+1}=squeeze(mnposition(ii(j),j,:));
end;