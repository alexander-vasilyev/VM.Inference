function [ CM,SM,C2M,S2M ] = radialmask( dim )
%RADIALMASK Summary of this function goes here
%   Detailed explanation goes here
ind=2*dim-1;

v=[1:1:ind];
v1= repmat(dim,1,ind);
v= v - v1;
v= repmat(v,1,ind);
v=vec2mat(v,ind);
v=v';
v=v;

w=[1:1:ind];
w1=repmat(dim,1,ind);
w=w-w1;
w= repmat(w,1,ind);
w= vec2mat(w,ind);


CM = repmat(1 , [ind ind]);
SM = repmat(1, [ind ind]);
C2M = repmat(1 , [ind ind]);
S2M = repmat(1, [ind ind]);
for i=1:ind
    for j=1:ind
        
        atn=atan2(v(i,j),w(i,j));
            
        if atn<0
            atn=atn+2*pi;
        end;
        
        CM(i,j)=CM(i,j)*cos(atn);
        SM(i,j)=SM(i,j)*sin(atn);
        C2M(i,j)=C2M(i,j)*cos(2*atn);
        S2M(i,j)=S2M(i,j)*sin(2*atn);
%         if atn<alpha && atn>beta
%             RM(i,j)=RM(i,j)*0.5;
%         end;
        
        
        
    end;  
end;

end

