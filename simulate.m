addpath('MDP');
addpath('convolutions')
addpath('interface');
%addpath('MFDFA');
% the application demonstrates the search task execution of observer 
% the scene state S, first coordinate - x, second - y
dim=32;
dimr=2*dim-1;
circ=false;
%choose to have multifractal analysis
mfdfa= false;
%choose to have Hurst exponent graphic
mfdfagr= 1;
%choose to calculate statistical properties
statistics= true;
%symmetric excecution function
alsym=true;
%choose heuristics
gr= false;
heur=true;
%dimensionality
N=1;
AN=2;
Radi=8;
%number of simulated patients
pats=40;
%initial probability
%P0 = repmat(0 , [128 128]);
if circ
    P0 = InitialP(dim);
else 
    P0 = repmat(1/(dim*dim) , [dim dim]);
end;

[CM,SM]=CircBasis(dim,N,AN,Radi); 
 

eplength=12;
epnum=1;


P=P0;

trajcl={};
meanc1={};
normcl={};
imbord=round(dim/4);
 
for viter=1:10
    %create random visibility map function
    hmeans=(randn(2*N*(AN+1),1))*30;
    %hmeans=0*hmeans;
    %  hmeans(2:2*N*(AN+1))=0*hmeans(2:2*N*(AN+1));
  %  hmeans(1)=-20*viter;
   % hmeans(1)=120;
    %save coefficients
    meanc1{end+1}=hmeans; 
    D2=IniHankel(hmeans,CM,SM,dim,N,AN);
    D2=abs(D2)+0.001;
    %D2=infval{viter};
    str='Initial visibility map';
    drawimage(D2,imbord,str,dim);
    
    rng('shuffle');

    eventtotc=zeros(dimr,dimr);
    

    iter=0;
    averreward=0;

    mln=round(sqrt(2)*dim);
    lhist= repmat( zeros, [1 mln]);
    anghist= repmat( zeros, [1 mln]);
    lenghist= repmat(zeros, [1 mln]);
    dirlenghist= repmat(zeros, [1 360]);
    dirhist= repmat(zeros, [1 360]);
    Xtraj = 0;
    Ytraj = 0;
    for repoch = 1:epnum
                                  
        movingaver= zeros(1,eplength);
        %average for each epoch
        %movingaver= 0;
        aviter=zeros(1,eplength);

        time=0;
        iter=1;

        RFun=Radialupdate2(0,dim,alsym,0);
        D1=D2.*D2;
        HFun=D1;
        HFun=rot90(rot90(D1));
        [Pfn,Hfn, fsize] = padarrays(P, HFun,'same');
        visdft=fft2(Hfn);
      
        for rep= 1:eplength
        disp(rep);
        S= targetloc(dim,circ);
        A= targetloc(dim,circ);
        % the initial probability - uniform
        P = P0;
        
        % total time of excecution
        time=0;
        %saccade sequence

        iter=0;

         % while (P(S(2),S(1))<0.5)
        while ( time<1500)

            % concatenation function for visibility (optimisation reasons)
            DC= conc(D2,A(1),A(2),dim);           
            % estimation of the observation vector - response from each location
            [O, prob] = observation(S,A,DC,dim,false);   

            % inference corresponding to observation 
            P = inference(P,O,A,DC);
            if gr
             str='Probability';
              drawimage(P,imbord,str,dim);
              pause(2);
            end
            % expected entropy estimation for each location
            H= conv_fft2(P,HFun,visdft,'same',dim,fsize); 

            %calculation of time normalized reward
            R= H;
            %R = reward(H,A,RFun,dim);
            if gr
             str='Reward function';
              drawimage(R,imbord,str,dim);
              pause(2);
            end
            % prior coordinates
            x0 = A(1);
            y0 = A(2);

            Xtraj(end+1)=x0;
            Ytraj(end+1)=y0;
            %estimation of best location to transfer the gaze
            [y,x] = find(R==max(R(:)));

%             FG= softmax(R,dim);
%             x=FG(1);
%             y=FG(2);
           
            if (length(x)>1)
%                   x(1)=S(1);                    
                x(2:length(x))=[];                  
            end;    
            if (length(y)>1)
%                  y(1)=S(2);
                y(2:length(y))=[];
            end;
            iter= iter+1;
            %stochastic saccade placement, intended saccade length:
            intsaclength= sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));                  
           % A=stoch(A,intsaclength,dim);
            A(1)=x;
            A(2)=y;

            %real saccade length
            saclength= sqrt((A(1)-x0)*(A(1)-x0)+(A(2)-y0)*(A(2)-y0)); 
            %if saclength>0
            eventtotc(dim+A(2)-y0,dim+A(1)-x0)=eventtotc(dim+A(2)-y0,dim+A(1)-x0)+1;
           % end;
            time = time + 200 + 0*saclength;


        end;
        Xtraj(end+1)=0;
        Ytraj(end+1)=0;
        movingaver(rep)=-time/10000;                                           
        aviter(rep)=iter;      


        end;
        revarray= movingaver;

        output= [mean(revarray)];
        disp(output);
        trajectory = [Xtraj Ytraj];

    end;    
 
        
    if statistics
%         figure
%         
%         plot(smooth(lhist/sum(lhist),'moving'));
%         
%         title('length distribution');
%         xlabel('pixels');
%         ylabel('frequency');



%         figure
% 
%         tails= (dirhist(1)+dirhist(360))/2;
%         dirhist(1)=tails;
%         dirhist(360)=tails;
% %         dirhist(1)=mean(dirhist(2:5));
% %         dirhist(360)=mean(dirhist(355:359));
%         
%        % plot(smooth(smooth(smooth(dirhist(1:360),'moving'),'moving'),'moving'));
%         polarplot(smooth(smooth(dirhist(1:360)/sum(dirhist),'moving'),'moving'),'-o');
%         title('direction distribution');
% %         xlabel('grad');
% %         ylabel('frequency');
%         
%         figure
%         
%         dirratio=dirlenghist./dirhist;
%         dirratio(1)=mean(dirratio);
%         dirratio(360)=mean(dirratio);
%         dirratio(361)=mean(dirratio);
%         dirratio(46)=mean(dirratio);
%         dirratio(91)=mean(dirratio);
%         dirratio(136)=mean(dirratio);
%         dirratio(181)=mean(dirratio);
%         dirratio(226)=mean(dirratio);
%         dirratio(271)=mean(dirratio);
%         dirratio(316)=mean(dirratio);
%         %plot(smooth(dirratio,'moving'));
%         %polarplot(smooth(smooth(dirratio(1:360),'moving'),'moving'),'-o');
%         title('length to horizontal angle');
%        
% 
%         figure
%         
%         plot(smooth(anghist,'moving'));
%         title('relative direction distribution');
%         xlabel('grad');
%         ylabel('frequency');
%         disp(sum(anghist(1:90))/sum(anghist));
%         
%         disp(sum(anghist(91:180))/sum(anghist));
%         
%         lenghist1= lenghist./anghist;
%         
%         figure
%         
%         plot(smooth(lenghist1,'moving'));
%         title('saccade length distribution to angle');
%         xlabel('grad');
%         ylabel('average length');
   str='Simulated distribution';
   drawimage(eventtotc,imbord,str,dim);
   end; 

   if mfdfa
        try                                    
                scmin=12;
                scmax=ceil(mean(aviter));
                scres=64;
                exponents=linspace(log2(scmin),log2(scmax),scres);
                scale=round(2.^exponents);
                q=linspace(-10,10,21);
                HqcorX=mfanalysis(Xtraj,scale,q,mfdfagr);
                HqcorY=mfanalysis(Ytraj,scale,q,mfdfagr);

         catch
                %disp('stack overflow in MF-DFA section');
         end;
    end;
        
    traj=[Xtraj;Ytraj];
    trajcl{end+1}=traj;
    
end;




