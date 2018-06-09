function [devi] = expln(devi,N,M)

for i=1:N
    for k=1:M
        if devi(i,k)<0
            devi(i,k)= exp(devi(i,k))+0.1;
        else
            devi(i,k)= log(devi(i,k)+1)+1.1;
        end;
        if devi(i,k)>4
            devi(i,k)=4;
        end;
    end;
end;


end