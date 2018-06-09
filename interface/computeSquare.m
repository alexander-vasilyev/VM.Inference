function y = computeSquare(xd,hj)
CM=hj{1};
SM=hj{2};

sz=size(CM);

AN=sz(1)-1;
N=sz(4);
dm=sz(2);
y=repmat(0,dm,dm);
for i=1:(AN+1)
    for j=1:N
        y=y+squeeze(CM(i,:,:,j)*xd((i-1)*N+j))+squeeze(SM(i,:,:,j)*xd(N*(AN+1)+(i-1)*N+j));
    end
end;    
