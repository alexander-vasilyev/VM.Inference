function [ B ] = inference(P,O,A,D )
% this function update the believe state according to observation vector O

D=D.*D;
W=O.*D;
P=P.*exp(W);
%calculation of numerator
% for rx = 1:128 
%     for ry = 1:128           
%         %Applying Bayes rule
%         P(rx, ry) = P(rx, ry)* exp(O(rx,ry)*D(abs(rx - A(1))+1, abs(ry - A(2))+1)*D(abs(rx - A(1))+1, abs(ry - A(2))+1));        
%     end
% end
%calculation of denominator
P(isnan(P))=1;
P(isinf(P))=1;
S = sum(sum(P));

%normalization of probability
P=P/S;

B = P;
end