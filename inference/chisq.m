function [ answ ] = chisq( h1,h2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


%# create an index of the "good" data points
goodIdx = h2>0; %# bin counts <0 are not good, either

sqh=(h1-h2).^2;
%d1 = sum(sum(h1(goodIdx).*log(h1(goodIdx)./h2(goodIdx))));
d2 = sum(sum(sqh(goodIdx)./h2(goodIdx)));

%# overwrite d only where we have actual data
%# the rest remains zero
answ = d2;