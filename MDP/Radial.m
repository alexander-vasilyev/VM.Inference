function [ R ] = Radial( A,B,RF,dim )
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
% v=[1:1:128];
% v1= repmat(A,1,128);
% v= v - v1;
% v= repmat(v,1,128);
% v=vec2mat(v,128);
% v= v.*v;
% 
% 
% w=[1:1:128];
% w1=repmat(B,1,128);
% w=w-w1;
% w= repmat(w,1,128);
% w= vec2mat(w,128);
% w=w.*w;
% 
% msum= w+v.';
% 
% msum= msum.^0.5;
% const= repmat(1, 128);
% R= const+ msum.*(pen)/200;
% R= R.^(-1);
ind=2*dim;
R=RF((dim+1)-A:ind-A, (dim+1)-B:ind-B);
% 
% colormap('hot');
% imagesc(R);
% set(gca,'YDir','normal')
% colorbar;
% drawnow;
% pause(100);

end

