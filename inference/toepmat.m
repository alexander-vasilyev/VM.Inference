function [ ftoep ] = toepmat( B,dim )
%TOEPLITZ Summary of this function goes here
%   Detailed explanation goes here

dimm=2*dim-1;

Bp=zeros(dimm);
Bp(dim:(dimm),1:dim)=B;
toep=zeros(dimm,dim,dimm);
zz=repmat(zeros, [dim-1 1]);
lb=2*dim;
for i=1:dimm
    col=Bp(lb-i,:);
    row=[col(1), zz'];
    ttoep=toeplitz(col',row);
    toep(:,:,i)=ttoep;
end;
ftoep=repmat(zeros,[dimm*dimm,dim*dim]);
for i=1:dimm
    for j=1:dim
        xind=1+(i-1)*dimm;
        xxind=xind+dimm-1;
        yind=1+(j-1)*dim;
        yyind=yind+dim-1;
        tind=0;
        if (i-j+1)>0
            tind=i-j+1;
        else
            tind=i-j+1+dimm;
        end;
        ftoep(xind:xxind,yind:yyind)=toep(:,:,tind);
    end;
end;


end

