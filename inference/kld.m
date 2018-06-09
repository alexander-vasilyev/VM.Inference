function [ answ ] = kld( h1,h2)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here


%# create an index of the "good" data points
goodIdx = h1>0 & h2>0; %# bin counts <0 are not good, either

%d1 = sum(sum(h1(goodIdx).*log(h1(goodIdx)./h2(goodIdx))));
d2 = sum(sum(h2(goodIdx).*log(h2(goodIdx)./h1(goodIdx))));

%# overwrite d only where we have actual data
%# the rest remains zero
answ = d2;
