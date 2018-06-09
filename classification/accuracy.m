function [ acc ] =  accuracy(scrs,amd,cont)

acc=0;
nrm=0;
for i=1:(amd)
     sc=scrs{i};
     acc=acc+sum(sc(1:(amd)));
end
for i=(amd+1):(amd+cont)
     sc=scrs{i};
     acc=acc+sum(sc((amd+1):(amd+cont)));
end
for i=1:(amd+cont)
    sc=scrs{i};
    nrm=nrm+sum(sc);
end

acc=acc/nrm;

end