function [ hist,corr ] = angcorr( HFun,eventtotc,eventfrec,jdist,dim,divi)

dimm=2*dim-1;
hist=repmat(0, [3 divi]);
quad=2*pi/divi;
for i=1:3
    if i==1
        inpmat=HFun;
    elseif i==2
        inpmat=eventtotc;
    elseif i==3
        inpmat=eventfrec;
    end;
    for j=1:dimm
        for k=1:dimm
            if (j-dim)*(j-dim)+(k-dim)*(k-dim)>0
                atn=atan2(j-dim,k-dim);            
                if atn<0
                atn=atn+2*pi;
                end;
                hist(i,floor(atn/quad)+1)=hist(i,floor(atn/quad)+1)+inpmat(j,k);                
            end;
        end;        
    end;
end;

for i=2:3
    hists=hist(i,:);
    hists=hists/sum(hists);
    hist(i,:)=hists;
end;
corr=1:6;
corrs=corrcoef(hist(1,:),hist(2,:));
corr(1)=corrs(1,2);
corrs=corrcoef(hist(2,:),hist(3,:));
corr(2)=corrs(1,2);
corrs=corrcoef(hist(1,:),hist(3,:));
corr(3)=corrs(1,2);
corrs=corrcoef(hist(1,:),jdist);
corr(4)=corrs(1,2);
corrs=corrcoef(hist(2,:),jdist);
corr(5)=corrs(1,2);
corrs=corrcoef(hist(3,:),jdist);
corr(6)=corrs(1,2);
end