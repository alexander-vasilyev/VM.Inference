function [pos ] = softmax( R,dim)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
L= exp(R*10000);
S=L/(sum(sum(L)));

S1= S(:)';
S2= cumsum(S1);
x = rand;
S3=S2-x;
S3(S3<0)=1;


[m, idx]= min(S3);


ver= fix(idx/dim)+1;
hor= rem(idx,dim)+1;

pos = [hor ver];
check=sum(sum(L));

if (isnan(check)||isinf(check))||((hor>dim)||(hor<1)||(ver>dim)||(ver<1))
   [x,y] = find(R==max(R(:)));
   pos= [x,y];
end;


% colormap('winter');
% imagesc(S);
% set(gca,'YDir','normal')
% colorbar;
% drawnow;
% pause(1);


end
