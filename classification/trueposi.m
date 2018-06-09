function [ acc ] =  trueposi(scrs,amd,cont)
acc=0;
nrm=0;
for i=(amd+1):(amd+cont)
     sc=scrs{i};
     acc=acc+sum(sc((amd+1):(amd+cont)));
end

for i=(amd+1):(amd+cont)
     sc=scrs{i};
     nrm=nrm+sum(sc);
end

acc=acc/nrm;

end