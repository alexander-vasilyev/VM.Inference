function [ Hcorr,Hshuf ] = mfanalysis( shifts, scale,scres, q, mfdfagr )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    
[Hq1,tq,hq,Dq,Fq] = MFDFA1(shifts,scale,q,1,0);
correction=log(Fq(1,1:scres));
figure
plot(correction);
shiftperm=shifts(randperm(length(shifts)));
[Hq2,tq,hq,Dq,Fq2] = MFDFA1(shiftperm,scale,q,1,0);
shiftperm1=shifts(randperm(length(shifts)));
[Hq3,tq,hq,Dq,Fq3] = MFDFA1(shiftperm1,scale,q,1,0);
shiftperm2=shifts(randperm(length(shifts)));
[Hq4,tq,hq,Dq,Fq4] = MFDFA1(shiftperm2,scale,q,1,0);

%  Faver=(Fq2+Fq3+Fq4)/3;
%  correction2=Faver(1,1:scres)/sum(Faver(1,1:scres));
% 
%  j=1;
%  for i=1:scres-1
%      if (Fq(1,i)/Fq(1,i+1)<0.1)||(Fq2(1,i)/Fq2(1,i+1)<0.1)||(Fq3(1,i)/Fq3(1,i+1)<0.1)||(Fq4(1,i)/Fq4(1,i+1)<0.1)
%      j=i;
%      end;
%  end;
% % 
% scmin=scale(j+1);
% scmax=scale(scres);
%                 
%                 %scmax=ceil(mean(aviter));
% scres=scres-j;
% exponents=linspace(log2(scmin),log2(scmax),scres);
% scale=round(2.^exponents);    
% 
% [Hq1,tq,hq,Dq,Fq] = MFDFA1(shifts,scale,q,1,mfdfagr );
% [Hq2,tq,hq,Dq,Fq2] = MFDFA1(shiftperm,scale,q,1,mfdfagr);
% [Hq3,tq,hq,Dq,Fq3] = MFDFA1(shiftperm1,scale,q,1,mfdfagr);
% [Hq4,tq,hq,Dq,Fq4] = MFDFA1(shiftperm2,scale,q,1,mfdfagr);


Hcorr=Hq1-(Hq2+Hq3+Hq4)/3;
Hshuf=(Hq2+Hq3+Hq4)/3;

figure
plot(Hcorr);
drawnow;
figure
plot(Hshuf);
drawnow;
end

