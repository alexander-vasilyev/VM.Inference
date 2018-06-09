function [deviup]= updevi(difmeansum,meanmat,timesum)

difsquare= difmeansum.*difmeansum;
meansquare= meanmat.*meanmat;
meanqube= meansquare.*meanmat;
sterm= difsquare-meansquare;
sterm=sterm./meanqube;      
answ=sterm*timesum';
answ(isnan(answ))=0;
answ(isinf(answ))=0;
deviup= answ;

end

