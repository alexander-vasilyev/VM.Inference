function [ R ] = reward( H,A, RFun,dim)

Rad= Radial(A(2),A(1),RFun,dim);

%for i = 1:128
%    for j= 1:128
        % estimating the rate of information gain for each of decisions: relocation gaze and hold on  
  %      if (i-A(1))*(i-A(1))+(j-A(2))*(j-A(2))>0
            % gaze relocation rate of information gain
 %           H(i,j) = H(i,j)/(200+ 0.1*pen*sqrt((i-A(1))*(i-A(1))+(j-A(2))*(j-A(2))));
   %     else 
            % hold on rate of information gain 
     %       H(i,j) = H(i,j)/50;
    %    end;
%    end
%end
H= H.*Rad;
R=H;

% % % % % % 
%  figure
%  colormap('hot');
%  imagesc(R);
%  set(gca,'YDir','normal')
%  colorbar;
%  drawnow;
%  pause(2);


end

