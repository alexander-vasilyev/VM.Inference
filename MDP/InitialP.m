function [ P0 ] = InitialP( dim )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
 P = repmat(1/(dim*dim) , [dim dim]);
 cent=dim/2+0.5;
for i = 1:dim
    for j = 1:dim
        if (sqrt((i-cent)*(i-cent)+(j-cent)*(j-cent)))<(dim/2)
            P(i,j)=0.01;
        end;
    end;
end;
S = sum(sum(P));
P=P./S;
P0=P;

% for i = 1:128
%     for j = 1:92
%         
%         P(i,j)=0.01;
%         
%     end;
% end;
% P0=P./sum(sum(P));

end

