function frt = hardmaxMC(x,dim)


[~,ii]=max(x);
freq=1:(dim*dim);
frt=histc(ii,freq);
frt=frt/sum(frt);
end