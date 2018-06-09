function [ ] = drawstats( sthist,slhist,mvhist)
 sthistmean=[];
 sthistvar=[];
 slhistmean=[];
 slhistvar=[];
 mnhistmean=[];
 mnhistvar=[];
 for i=1:length(sthist)
     sthistmean(end+1)=mean(sthist{i});
     sthistvar(end+1)=var(sthist{i});
     slhistmean(end+1)=mean(slhist{i});
     slhistvar(end+1)=var(slhist{i});
     mnhistmean(end+1)=mean(mvhist{i});
     mnhistvar(end+1)=var(mvhist{i});
 end;
   figure;
   errorbar(sthistmean,sqrt(sthistvar));
   title('fixation duration');
   drawnow;
   figure;
   errorbar(slhistmean,sqrt(slhistvar));
   title('saccade length');
   drawnow;
   figure
   errorbar(mnhistmean,sqrt(mnhistvar));
   title('execution time');
   drawnow;
    
end