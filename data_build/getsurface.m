function elementSurface=getsurface(min_boundary,max_boundary,elementSurface,nodes)
% global nodes
    elementSurface(:,2:4) = elementSurface;
    elementSurface(:,1) = 1;
    for i = 1 : length(elementSurface(:,1)) % 找到小鼠上下底面，并将其设为内部边界
         a = elementSurface(i,2:4);
         node = nodes(a,3);
         if mean(node)<min_boundary || mean(node)>max_boundary
             elementSurface(i,1) = 0; % 内部边界
         end
     end
     flagSurface = find(elementSurface(:,1)==1); % 外边界
     elementSurface = elementSurface(flagSurface,:);