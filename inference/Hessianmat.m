function [ Hessian] = Hessianmat(linegradientsum,lpsum,defmeansum,N,eplength)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% gaussian approximation of visibility map
probone=linegradientsum+lpsum;
probtwo=-linegradientsum+lpsum;
Hessian=repmat(0,5*N);
Hesscomp=zeros(5*N,5*N,eplength);
diagh=diag(repmat(-1,[1,5*N]));
baseline=0;
div=0;
for i=1:eplength
    Hesscomp(:,:,i)=defmeansum(:,i)*defmeansum(:,i)'+diagh;    
end;
for i=1:eplength
    hsc=Hesscomp(:,:,i);
    baseline=baseline+sum(sum(hsc.*hsc))*(probone(i)+probtwo(i));
    div=div+sum(sum(hsc.*hsc));
end;
baseline=baseline/(2*div);
for i=1:eplength
    Hessian=Hessian+Hesscomp(:,:,i)*(probone(i)+probtwo(i)-2*baseline);
end;
Hessian=Hessian/(2*eplength);
%used for observation and inference
end