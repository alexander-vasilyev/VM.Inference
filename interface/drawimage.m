function [  ] = drawimage( IM, imbord,str,dim)
    figure;
    colormap('hot');
    
    imagesc(IM);   
    set(gca,'YDir','normal')
    title(str);
    colorbar;
    drawnow;    
end