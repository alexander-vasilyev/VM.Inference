function [ dhf, dhf1 ] = distanceVM( HFun,HFunt,Radi)
 HFuntg=imgaussfilt(HFunt,4);
 HFung=imgaussfilt(HFun,4);
 [xh1,yh1] = find(HFuntg==max(HFuntg(:)));
 [xh,yh]= find(HFung==max(HFung(:)));
 
 H1=HFun(xh-Radi:xh+Radi,yh-Radi:yh+Radi);
 H2=HFunt(xh1-Radi:xh1+Radi,yh1-Radi:yh1+Radi);
 sfn=sqrt(sum(sum(H1.*H1)));
 sfn1=sqrt(sum(sum(H2.*H2)));
 dhf=sum(sum((H2/sfn1-H1/sfn).*(H2/sfn1-H1/sfn)));
 dhf1=sum(sum((H2-H1).*(H2-H1)))/(sfn1*sfn1);

end