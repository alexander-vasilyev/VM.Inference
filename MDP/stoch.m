function [A]= stoch(A,saclength,dim)
Xpos= 0;
Ypos= 0;
while ((round(Xpos)>dim || round(Xpos)<1)||(round(Ypos)>dim || round(Ypos)<1))
    Xpos = A(1) + randn*(0*saclength/10+3);
    Ypos = A(2) + randn*(0*saclength/10+3);
end;
A(1)= round(Xpos);
A(2)= round(Ypos);
end