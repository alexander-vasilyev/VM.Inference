function [ D ] = conc( D0,A,B,dim )

ind=2*dim;
 D=D0(((dim+1)-B):(ind-B),((dim+1)-A):(ind-A));


end

