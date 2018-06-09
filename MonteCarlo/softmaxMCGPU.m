function gx = softmaxMCGPU(x)
% softmax mapping, i.e.: g(x) = exp(x)./sum(exp(x))
% function gx = softmax(x,beta)

%x = x-repmat(max(x),size(x,1),1);

ex=exp(x);
ssex=sum(ex);
gx=bsxfun(@rdivide,ex,ssex);
end
