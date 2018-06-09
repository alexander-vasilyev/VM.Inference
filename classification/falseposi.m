function [ acc ] = falseposi(scrs,amd,cont)

acc=0;
norm=0;

for i=1:(amd)
sc=scrs{i};
acc=acc+sum(sc((amd+1):(amd+cont)));
norm=norm+sum(sc);
end

acc=acc/norm;
end