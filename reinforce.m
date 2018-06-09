addpath('C:/Users/av306/Desktop/sasha/multifractal paper/mf-dfa-infomax/code/MFDFA')
addpath('C:/Users/av306/Desktop/sasha/multifractal paper/mf-dfa-infomax/code/MDP')

% the application demonstrates the search task execution of observer 
% the scene state S, first coordinate - x, second - y
S = [10,10];
%choose to have multifractal analysis
mfdfa= false;
%choose to have Hurst exponent graphic
mfdfagr= 0;
%choose to calculate statistical properties
statistics= true;
%dimensionality
N=10;
%initial probability
P0 = repmat(1/(128*128) , [128 128]);
%P0= InitialP(P0);
gausmat=gauss;
G=RadialBasis1(1,N);
F=RadialBasis1(0,N);
% means= (rand(N,1) -0.5)*10;
% means(1:20)=means0;

% means= rand(N,1) -0.5;
%means=load('radial.reinforce.txt');
hmeans= (rand(N,1) - 0.5)*10;
%hmeans=load('smoothing.reinf.0.15.txt');
%means=hank*1000;

%hmeans=hank/30;
eplength=1000;
epnum=400;

% bessfft=zeros(255,255,N);
% for i=1:N
%     F1=F(:,:,i);
%     bessfft(:,:,i)=Dfft(P0, F1);    
% end;


for pen=1:1
    %average value 
    %histogram for length-angle distribution
    RFun= IniHankel(G, means, N);
    %RFun=ones(255);
    HFun= IniHankel(F, hmeans, N);
    figure
    plot(diag(RFun));
    hold on;
    plot(diag(HFun));
    drawnow;
    %statistics
    lhist= repmat( zeros, [1 181]);
    anghist= repmat( zeros, [1 181]);
    lenghist= repmat(zeros, [1 181]);
    dirhist= repmat(zeros, [1 181]);
    
    for V=10:10
        rng('shuffle');
        %average Hurst
        avhurst=[0,0,0];
        hurcor=0;
        % Visibility map
        g=FPOC(0.2, V/100);
        [D1, D2] = Visibility(g);
        A = [64, 64];
        % dfa extended trajectory
        
       
        iter=0;
        averreward=0;
        meanarray= zeros(N,pen,epnum);
        hmeanarray= zeros(N,pen,epnum);
        for repoch = 1:epnum
            
            
            Xtraj = A(1);
            Ytraj = A(2);
           
            radgradientsum= zeros(N,eplength);
            smogradientsum= zeros(N,eplength);
            movingaver= zeros(1,eplength);
            %average for each epoch
            %movingaver= 0;
            aviter=zeros(1,eplength);
            %estimated gradient
            gradientmat= zeros(182 , 1);
            time=0;
            iter=1;
            RFun= IniHankel(G, means,N);
            %RFun=ones(255);
            HFun=IniHankel(F,hmeans,N);
            HFun=D1;
            visdft= Dfft(P0,HFun);
            for rep= 1:eplength
            
            %start of gradient
            %MP=targetloc(rep);
            S= targetloc(rep);
            A= targetloc(rep);
            A=[64,64];
            % the initial probability - uniform
            P = P0;
            % vector of first location corresponding to the center of the screen
            
            % vector of cell with maximal probability of target
            
            % X and Y coordinates of trajectory 
            Xloc = 0;
            Yloc = 0;
            
            
            % total time of excecution
            time=0;
            % value function
            
            % information
            
            radgradient=zeros(N,1);
            smogradient=zeros(N,1);
            
            %saccade sequence
            %I1=I0;
            
            
            
            
            
            iter=0;
                %while (((A(1)-S(1))*(A(1)-S(1))+(A(2)-S(2))*(A(2)-S(2)))>6)
                while(P(S(1),S(2))<0.5)
                %while ( time<10000)
                %I= Information(P);
                
                % concatenation function for visibility (optimisation reasons)
                DC= conc(D2,A(1),A(2));           
                % estimation of the observation vector - response from each location
                O = observation(S,A,DC);   
                %prior distribution

                % inference corresponding to observation 
                P = inference(P,O,A,DC);
                % expected entropy estimation for each location
                H = entropy(P,D1,visdft);               
                %calculation of time normalized reward
                R = reward(H,A,RFun);
                %soft= softmax(R);
                % prior coordinates
                x0 = A(1);
                y0 = A(2);
                %estimation of best location to transfer the gaze
                [x,y] = find(R==max(R(:)));
                %[MP(1),MP(2)]=find(P==max(P(:)));
                if (length(x)>1)
 %                   x(1)=S(1);                    
                    x(2:length(x))=[];                  
                end;    
                if (length(y)>1)
  %                  y(1)=S(2);
                    y(2:length(y))=[];
                end;
                %gradient estimation               
                
                    
                %smogradient= smogradient + gradientsmo(H,P,x0,y0,x,y,1,F,N,RFun);
                %radgradient= radgradient + gradientrad(H,x0,y0,x,y,100,RFun,G,N,gausmat); 
                iter= iter+1;
                %estimation of location with maximal probaility of target (equal only
                %in MAP case                      
                
                %dislocation of the gaze to this location
                
                A(1) = x;
                A(2) = y;
                

                %dislocation of attention
                
                %saving saccade sequence
                
                
                %stochastic saccade placement
                saclength= sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)); 
%                if saclength>5
                    A(1)=x;
                    A(2)=y;                    
%                 else
%                     A(1)=x0;
%                     A(2)=y0;                  
%                 end;
                Xloc(end+1)= A(1)-x0;
                Yloc(end+1)= A(2)-y0;
              %  A=stoch(A,saclength);
                saclength=sqrt((A(1)-x0)*(A(1)-x0)+(A(2)-y0)*(A(2)-y0));
                %radgradient= radgradient + gradientrad(H,x0,y0,A(1),A(2),100,RFun,G,N,gausmat); 
                radgradient= radgradient + gradientrad1(H,x0,y0,A(1),A(2),100,RFun,G,N);
                
                %                 if mfdfa
%                 %saving for extended trajectory
%                     if saclength<2
%                         for i= 1:4
%                         	Xtraj(end+1)= 0;
%                             Ytraj(end+1)= 0;
%                         end;
%                         Xtraj(end+1)=(x-x0);
%                         Ytraj(end+1)=(y-y0);
%                     else 
%                         for i= 1:4
%                             Xtraj(end+1)= 0;
%                             Ytraj(end+1)= 0;
%                         end;
%                         for i=1:ceil(saclength/40)
%                             Xtraj(end+1)=(x-x0)/(ceil(saclength/40));
%                             Ytraj(end+1)= (y-y0)/(ceil(saclength/40));
%                         end;
%                     end;
%                 end;
 
                 % P = (4*P + circshift(P, [-1 0 0]) + circshift(P, [1 0 0]) + circshift(P, [0 -1 0]) + circshift(P, [0 1 0]))/(4+4);

                % time estimation
%                 if (x0-x)*(x0-x)+(y0-y)*(y0-y)<25
%                     time = time + 50;
%                 else
                    time = time + 200 + 0*saclength;
%                 end;
                % estimation of statistal properties
%                 if statistics
%                     if saclength>5
%                         lhist(floor(saclength)+1)=lhist(floor(saclength)+1) +1;
%                         dirang= acosd((x-x0)/sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0)));
%                         dirhist(floor(real(dirang))+1)=dirhist(floor(real(dirang))+1)+1;
%                         
%                     end;
%                 
%                 
%                     if ((length(Xloc)>2 && Xloc(end)*Yloc(end)~=0) && Xloc(end-1)*Yloc(end-1)~=0)
%                         sacangle= acosd((Xloc(end)*Xloc(end-1)+Yloc(end)*Yloc(end-1))/(sqrt(Xloc(end)*Xloc(end)+Yloc(end)*Yloc(end))*sqrt(Xloc(end-1)*Xloc(end-1)+Yloc(end-1)*Yloc(end-1))));
%                         anghist(floor(real(sacangle))+1)=anghist(floor(real(sacangle))+1) +1;
%                         lenghist(floor(real(sacangle))+1)=lenghist(floor(real(sacangle))+1)+saclength/100;
%                     end;
%                 end;  

                end;
            movingaver(rep)=-time/10000;                                           
            aviter(rep)=iter;      
            
            radgradientsum(:,rep)=radgradient;    
            smogradientsum(:,rep)=smogradient;

            end;
            
            
            
            revarray= movingaver;
%             movingaver1=(movingaver(1:100)-mean(movingaver(1:100)));
%             movingaver2=(movingaver(101:200)-mean(movingaver(101:200)));
%             movingaver3=(movingaver(201:300)-mean(movingaver(201:300)));
%             movingaver4=(movingaver(301:400)-mean(movingaver(301:400)));
%             
%             normmoving= [movingaver1, movingaver2, movingaver3, movingaver4];
%             %normmoving= normmoving./aviter;
%             normmoving= movingaver;
%             normmoving= -normmoving/1000;
            %rmss= rms(gradiaver.*normmoving)/length(normmoving);
            %rmsrew= rms(movingaver)/length(movingaver);
            
            %GPOMDP
            linequad=radgradientsum.*radgradientsum;
            baseline=linequad*revarray';
            if sum(sum(linequad))>0
                baseline=baseline./(sum(linequad,2));
                baseline(1)=0;
                rbessres=radgradientsum*revarray'-sum(radgradientsum,2).*baseline;
            else
                rbessres=zeros(N,1);
            end;
            %GPOMDP END
            
%             slinequad=smogradientsum.*smogradientsum;
%             sbaseline=slinequad*revarray';
%             if sum(sum(slinequad))>0
%                 sbaseline=sbaseline./(sum(slinequad,2));
%                 sbessres=smogradientsum*revarray'-sum(smogradientsum,2).*sbaseline;
%             else
%                 sbessres=zeros(N,1);
%             end;
            %RFun=RFun+Radialupdate(gradres);
            
            %ACTOR-CRITIC 
%             mat1= zeros(45);
%             mat2= zeros(45);
%             for i= 1:400
%                 mat1=mat1+radgradientsum(:,i)*radgradientsum(:,i)';  
%             end;
%             fisher=mat1;
%             ellig= sum(radgradientsum,2);
%             diffisher= ellig*ellig';
%             Q=(1+((ellig')*inv(400*fisher-diffisher))*ellig)/400;
%             vanilla= radgradientsum*revarray';
%             
%             base=Q*(sum(revarray)-(ellig'*inv(fisher))*vanilla);
%             bessres= inv(fisher)*(vanilla-ellig*base);
            %ACTOR-CRITIC END
            
            %UPDATE OF PARAMETERS
            meanarray(:,pen,repoch)=means;
            hmeanarray(:,pen,repoch)=hmeans;
           
            means=means+rbessres;
           % hmeans=hmeans+sbessres/1000;
            
            RFun0=RFun;
            RFun= IniHankel(G, means,N);
            DRFun=RFun-RFun0;
            grd=sum(sum(DRFun.*DRFun));
            HFun0=HFun;
            
            HFun= IniHankel(F, hmeans,N);
            
            DHfun=HFun-HFun0;
            grd1=sum(sum(DHfun.*DHfun));
         
           
            
            
            figure
            plot(diag(RFun));
            hold on;
            plot(diag(HFun));
            drawnow;
            %UPDATE END
            
                                    
            %output of current penalty and statistical properties
            output= [mean(revarray)];
            disp(output);
        end;    

         
        
    end;
    


end;
% figure
%         
%         l1=plot(smooth(anghist(1,:)/sum(anghist(1,:)),'moving'));
%        
%         hold on;
%         l2=plot(smooth(anghist(2,:)/sum(anghist(2,:)),'moving'));
%        
%         hold on;
%         l3=plot(smooth(anghist(3,:)/sum(anghist(3,:)),'moving'));
%         hold on;
%      
%         legend('infomax greedy','learned kernel','infomax rate','Location','northeast');
%         title('relative angle distribution');
%         xlabel('grad');
%         ylabel('frequency');

        



