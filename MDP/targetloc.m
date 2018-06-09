function [S]= targetloc (dim,circ)

if ~circ
x= floor((dim)*rand);
y= floor((dim)*rand);
else 
x= floor((dim)*rand);
y= floor((dim)*rand);    
    while ((x-dim/2)*(x-dim/2)+(y-dim/2)*(y-dim/2))>(dim*dim/4)
        x= floor((dim)*rand);
        y= floor((dim)*rand);
    end;
end;
S=[x+1,y+1];



end