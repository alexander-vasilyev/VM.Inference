addpath('MDP');
addpath('convolutions')
addpath('MFDFA');
%addpath('MFDFA');
% the application demonstrates the search task execution of observer 
% the scene state S, first coordinate - x, second - y
dim=128;
dimr=2*dim-1;
circ=false;
%choose to have multifractal analysis
mfdfa= true;
%choose to have Hurst exponent graphic
mfdfagr= 1;
%choose to calculate statistical properties
statistics= false;
%symmetric excecution function
alsym=true;
%penalty for small saccades
shsac=false;
%scale
scaleg=6;
%P0 = repmat(1/(128*128) , [128 128]);
if circ
    P0 = InitialP(dim);
else 
    P0 = repmat(1/(dim*dim) , [dim dim]);
end;

eplength=200;
epnum=1;


P=P0;

trajcl={};
meanc1={};
normcl={};
imbord=round(dim/4);
for viter=1:1
    
    rng('shuffle');

    eventtotc=zeros(dimr,dimr);
    g=FPOC(0.2,0.1+0.05*(viter-1),dim,scaleg); 
    [D1,D2]=Visibility(g,dim);
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
                                  
        Xloc = 0;
        Yloc = 0;

        movingaver= zeros(1,eplength);
        %average for each epoch
        %movingaver= 0;
        aviter=zeros(1,eplength);

        time=0;
        iter=1;


        RFun=Radialupdate2(0,dim,alsym,shsac);
        D1=D2.*D2;
        HFun=D1;
        HFun=rot90(rot90(D1));
        [Pfn,Hfn, fsize] = padarrays(P, HFun,'same');
        visdft=fft2(Hfn);

        for rep= 1:ceil(eplength/viter)

        S= targetloc(dim,circ);
        A= targetloc(dim,circ);
        A=[64,64];
        % the initial probability - uniform
        P = P0;
        disp(rep); 

        % total time of excecution
        time=0;


        %saccade sequence

        iter=0;

            while(P(S(1),S(2))<0.5)
        %while ( time<6000)

            % concatenation function for visibility (optimisation reasons)
            DC= conc(D2,A(1),A(2),dim);           
            % estimation of the observation vector - response from each location
            [O, prob] = observation(S,A,DC,dim,true);   

            % inference corresponding to observation 
            P = inference(P,O,A,DC);
            % expected entropy estimation for each location
            H= conv_fft2(P,HFun,visdft,'same'); 

            %calculation of time normalized reward
            R = reward(H,A,RFun,dim);

            % prior coordinates
            x0 = A(1);
            y0 = A(2);


            %estimation of best location to transfer the gaze
            [x,y] = find(R==max(R(:)));

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
            %A=stoch(A,intsaclength,dim);
            A(1)=x;
            A(2)=y;
            %real saccade length
            saclength= sqrt((A(1)-x0)*(A(1)-x0)+(A(2)-y0)*(A(2)-y0)); 
           % if saclength>1
            eventtotc(dim+A(1)-x0,dim+A(2)-y0)=eventtotc(dim+A(1)-x0,dim+A(2)-y0)+1;
           % end;
            %saving for extended trajectory
            Xloc(end+1)=A(1);
            Yloc(end+1)=A(2);
%                Xtraj(end+1)=abs(A(1)-x0);
%                Ytraj(end+1)=abs(A(2)-y0);
            if saclength<5
                for i= 1:10
                    Xtraj(end+1)= 0;
                    Ytraj(end+1)= 0;
                end;
                Xtraj(end+1)=(A(1)-x0);
                Ytraj(end+1)=(A(2)-y0);
            else 
                for i= 1:10
                    Xtraj(end+1)= 0;
                    Ytraj(end+1)= 0;
                end;
                for i=1:round(saclength/5)
                    Xtraj(end+1)=(A(1)-x0)/round(saclength/5);
                    Ytraj(end+1)= (A(2)-y0)/round(saclength/5);
                end;
            end;


%             if saclength<10
%                 for i= 1:20
%                     Xtraj(end+1)= x0;
%                     Ytraj(end+1)= y0;
%                 end;                     
%             else 
%                 for i= 1:20
%                     Xtraj(end+1)= x0;
%                     Ytraj(end+1)= y0;
%                 end;
%                 for i=1:ceil(saclength/10)
%                     Xtraj(end+1)=x0+(i-1)*(x-x0)/(ceil(saclength/10));
%                     Ytraj(end+1)=y0+(i-1)*(y-y0)/(ceil(saclength/10));
%                 end;
%             end;                   




                 time = time + 200 + 0*saclength;

            if statistics
                if (saclength>0)
                    lhist(floor(saclength)+1)=lhist(floor(saclength)+1) +1;

                    dirang= atan2((y-y0),(x-x0))*180/pi;
                    if dirang<0 
                    dirang= dirang+360;
                    end;
                    if dirang==0
                    dirang=0.001;
                    end;


                    dirhist(ceil(real(dirang)))=dirhist(ceil(real(dirang)))+1;                       
                    dirlenghist(ceil(real(dirang)))=dirlenghist(ceil(real(dirang)))+saclength; 
                end;              
                if ((length(Xloc)>2 && Xloc(end)*Yloc(end)~=0) && Xloc(end-1)*Yloc(end-1)~=0)
                    sacangle= acosd((Xloc(end)*Xloc(end-1)+Yloc(end)*Yloc(end-1))/(sqrt(Xloc(end)*Xloc(end)+Yloc(end)*Yloc(end))*sqrt(Xloc(end-1)*Xloc(end-1)+Yloc(end-1)*Yloc(end-1))));

                    anghist(floor(real(sacangle))+1)=anghist(floor(real(sacangle))+1) +1;
                    lenghist(floor(real(sacangle))+1)=lenghist(floor(real(sacangle))+1)+saclength;
                end;
            end;  

        end;

        movingaver(rep)=-time/10000;                                           
        aviter(rep)=iter;      


        end;
        revarray= movingaver;
 
        trajectory = [Xtraj Ytraj];

    end;    
 
        
        if statistics
        figure
        
        plot(smooth(lhist/sum(lhist),'moving'));
        
        title('length distribution');
        xlabel('pixels');
        ylabel('frequency');

        figure

        tails= (dirhist(1)+dirhist(360))/2;
        dirhist(1)=tails;
        dirhist(360)=tails;
%         dirhist(1)=mean(dirhist(2:5));
%         dirhist(360)=mean(dirhist(355:359));
        
       % plot(smooth(smooth(smooth(dirhist(1:360),'moving'),'moving'),'moving'));
        polarplot(smooth(smooth(dirhist(1:360)/sum(dirhist),'moving'),'moving'),'-o');
        title('direction distribution');
%         xlabel('grad');
%         ylabel('frequency');
        
        lenghist1= lenghist./anghist;
        
        figure
        
        plot(smooth(lenghist1,'moving'));
        title('saccade length distribution to angle');
        xlabel('grad');
        ylabel('average length');
        
        figure;
        imagesc(transpose((eventtotc)));
        set(gca,'YDir','normal')
        drawnow;
        end; 

   if mfdfa
        try                                    
                scmin=64;
                scmax=128;
                
                %scmax=ceil(mean(aviter));
                scres=128;
                exponents=linspace(log2(scmin),log2(scmax),scres);
                scale=round(2.^exponents);
                q=linspace(-10,10,21);
                xtrn=randn(length(Xtraj),1);
                shifts=abs(Xtraj+0.0*xtrn');
                    

                [Hq1,tq,hq,Dq,Fq] = MFDFA1(shifts,scale,q,1,1);
                 shiftperm=shifts(randperm(length(shifts)));
                 [Hq2,tq,hq,Dq,Fq2] = MFDFA1(shiftperm,scale,q,1,mfdfagr);
                shiftperm1=shifts(randperm(length(shifts)));
                [Hq3,tq,hq,Dq,Fq3] = MFDFA1(shiftperm1,scale,q,1,mfdfagr);
                shiftperm2=shifts(randperm(length(shifts)));
                [Hq4,tq,hq,Dq,Fq4] = MFDFA1(shiftperm2,scale,q,1,mfdfagr);
                Hcorr=Hq1-(Hq2+Hq3+Hq4)/3;
                Hshuf=(Hq2+Hq3+Hq4)/3;

                figure
                plot(Hcorr);
                drawnow;
                figure
                plot(Hshuf);
                drawnow;
                
%                 correction=log(Fq(1,1:scres));
%                 figure
%                 plot(correction);
%                [hcorr,hshuf]=mfanalysis(shifts,scale,scres,q,mfdfagr);

         catch
                %disp('stack overflow in MF-DFA section');
         end;
    end;
        
    traj=[Xtraj;Ytraj];
    trajcl{end+1}=traj;
    
end;