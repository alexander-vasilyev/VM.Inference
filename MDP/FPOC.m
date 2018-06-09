function [ fpoc] = FPOC( ctar,cnois,dim,scale)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% gaussian approximation of visibility map
mln=round(sqrt(2)*dim);
f=zeros(1,mln);
s=zeros(1,mln);
ct=zeros(1,mln);
slope=zeros(1,mln);
inter=zeros(1,mln);
invct=zeros(1,mln);
for i=1:mln
    s(i)=2.8*((i-1)/scale)/((i-1)/scale+0.8) +2;
end;

for i=1:mln
    slope(i)=(10^(0.21*(i-1)/scale))/1.66;
end;

for i=1:mln
    inter(i)=(10^(0.35*(i-1)/scale))/750;
end;

for i=1:mln
    ct(i)=cnois*slope(i)+inter(i);
end;

for i=1:mln
    invct(i)=ctar/ct(i);
end;


invct=invct.^s;




for i=1:mln
    f(i)=0.5+0.5*(1- 2.71^(-invct(i)));
end;


fpoc= sqrt(2)*icdf('Normal',f,0,1);



%used for observation and inference


end

