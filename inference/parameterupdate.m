function [ mixing, hmeans,updatematrix,devi] = parameterupdate(mixing,hmeans,L,M,vlpsum,mixgradientsum,linegradientsum,devi,baseline,step,defmeansum,eplength)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% gaussian approximation of visibility map
mixing0=mixing;
updatematrix=defmeansum*linegradientsum';

for j=1:L
    for k=1:eplength
        mg=mixgradientsum(j,k);
        if mixing(j,mg)<1
            mixing(j,mg)=mixing(j,mg)+(vlpsum(k)-baseline)*step/(M);
        end;
        if abs(devi(j,mg))>0
            defi=(defmeansum(j,k)^2-devi(j,mg)^2)/devi(j,mg);
            devi(j,mg)=devi(j,mg)+(vlpsum(k)-baseline)*step*defi/(10*M);
        end;        
        hmeans(j,mg)=hmeans(j,mg)+updatematrix(j)*step;
    end;
end;

for j=1:L    
    mix=mixing(j,:);
    mix(mix<0)=0;
    if sum(mix)>0
        mix=mix/sum(mix);
    else 
        mix=mixing0(j,:);
    end;
    mixing(j,:)=mix;
end;
%used for observation and inference
end