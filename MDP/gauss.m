function[gaussum]=gauss(sigma)

v=[1:1:255];
v1= repmat(128,1,255);
v= v - v1;
v= repmat(v,1,255);
v=vec2mat(v,255);
v= v.*v;


w=[1:1:255];
w1=repmat(128,1,255);
w=w-w1;
w= repmat(w,1,255);
w= vec2mat(w,255);
w=w.*w;

msum= w+v.';

msum= msum.^0.5;
gaussum= arrayfun(@(x) exp(-x*x/(2*sigma*sigma))/(sqrt(2*3.14)), msum);


end