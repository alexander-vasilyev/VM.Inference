function [ O, lprob ] = observation(S,A,D,dim,tar )
%UNTITLED4 Summary of this function goes here
%   This function returns the vector of observation depending on the position
%  of gaze and the scene configuration

 
% number of objects
% definition of observation vector
O = repmat(-0.5 , [dim dim]);
% deterministic part of signal
if tar      
    O(S(2),S(1)) = 0.5;
end;
%colormap('hot');
%imagesc(O);
%colorbar;
% internal noise




% generation of gaussian noise at each location
% estimation of visibility map function D


%colormap('hot');
%imagesc(V);
%colorbar;


mrandn = randn(dim);

D=D.^(-1);
noise= mrandn.*D;
O=O+noise;
% srandn=mrandn.*mrandn/2;
% lprob=0;
% 
% mbrandn=randn(256);
% mbrandn=mbrandn.*mbrandn/2;
% mbrandn(65:192,65:192)=srandn;
% for rx = 1:256 
%    for ry = 1:256           
%         if ((rx-A(1)-64)*(rx-A(1)-64)+(ry-A(2)-64)*(ry-A(2)-64))<128
%         lprob = lprob +  mbrandn(rx,ry); 
%         end;
%    end
% end
% lprob=lprob/sqrt(2*pi);
lprob=0;


end

