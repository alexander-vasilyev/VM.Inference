function [ modal, hmean,dev] = selectmodal(mixing,hmeans,devi,L)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% gaussian approximation of visibility map
M=zeros(L,1);
dev=zeros(L,1);
for i=1:L
X=rand;
cm=cumsum(mixing(L,:));
sm=cm-X;
sm(sm<0)=1;
[m, M(i)]=min(sm);
end;
hmean=zeros(L,1);
for i=1:L
hmean(i)=hmeans(i,M(i));
dev(i)=devi(i,M(i));
end;
modal=M;

%used for observation and inference
end