function [D1, D2] = Visibility( g,dim )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% gaussian approximation of visibility map
ind=2*dim-1;
v=[1:1:ind];
v1= repmat(dim,1,ind);
v= v - v1;
v= repmat(v,1,ind);
v=vec2mat(v,ind);
v= v.*v;

w=[1:1:ind];
w1=repmat(dim,1,ind);
w=w-w1;
w= repmat(w,1,ind);
w= vec2mat(w,ind);
w=w.*w;

msum= w+ v.'; %0.75*v.';

msum= msum.^0.5;
msum1=msum;
msum1(msum1<10)=0.1;
msum1(msum1>9)=1;




D1= repmat (zeros, [ind ind]);



for i= 1:ind
    for j= 1:ind        
        D1(i,j)= 0.001+g(ceil(msum(i,j)+0.01))*g(ceil(msum(i,j)+0.01)); 
%            if (i>128)%% &&(j<128)
%                D1(i,j)=D1(i,j)*0.49;
%            end;
    end;
%     for j=129:255
%         D1(i,j)=D1(i,j)*0.01;
%     end;
end;


%used for observation and inference
D2= repmat (zeros, [ind ind]);

for i= 1:ind
    for j= 1:ind        
        D2(i,j)=0.001+ g(ceil(msum(i,j)+0.01));
% % 
%          if (i>128)%% &&(j<128)
%              D2(i,j)=D2(i,j)*0.7;
%          end;
    end;
%     for j= 129:255
%         D2(i,j)=D2(i,j)*0.1;
%     end;
end;
% figure
% colormap('hot');
% imagesc(D2);
% %set(gca,'YDir','normal')
% colorbar;
% drawnow;
% pause(2);


end

