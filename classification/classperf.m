function [ score ] =  classperf(probseq,div)
divprobseq={};
sclist={};

for j=1:500
    divprobseq={};
for i=1:length(probseq)
    eprob=probseq{i};
    difprseq=[];
    for k=1:(length(eprob)-1)
        difprseq(end+1)=eprob(k+1)-eprob(k);
    end
    
    comprobs=[];
    probs=0;
    indmat=randperm(length(difprseq));
    
    for k=1:length(difprseq)
          probs=probs+difprseq(indmat(k));
          if rem(k,div)==0
          comprobs(end+1)=probs;
          probs=0;
          end;
    end
    divprobseq{end+1}=comprobs;
end

divprob=cell2mat(divprobseq);
kiter=length(divprob)/length(divprobseq);
divprob=vec2mat(divprob, kiter);
score=0*(1:length(divprobseq));
for k=1:kiter
     kscore=divprob(:,k);
     [M,I]=max(kscore);
     score(I)=score(I)+1;
end;
score=score/sum(score);
sclist{end+1}=score;

end;
score=0*(1:length(divprobseq));
for j=1:length(sclist)
    score=score+sclist{j};
end;
score=score/length(sclist);
end

