function [ elementInside , elementSurface , edgeMatrix ] = pretreatment( elementInside , elementSurface )
%PRETREATMENT Summary of this function goes here
%  预处理elementInside
elementInside=cat(2,elementInside,zeros(length(elementInside(:,1)),9));
edgeMatrix=[];
num1=7;
for i=1:length(elementInside(:,1))
    
    if mod(i,500)==0
        a = sprintf('Now at %d step',i); 
        disp([a,'  :  ',num2str(toc)])
    end
    
    vector1=elementInside(i,2:5);
    p1=[1 2;1 3;1 4;2 3;2 4;3 4];    
    if i==1
        edgeMatrix(1:6,2:3)=vector1(p1(1:6,:));
        elementInside(i,6:11)=1:6;
    else
        for ii=1:6
            flag=1;
%             for iii=1:length(edgeMatrix(:,1))
%                 if sort(vector1(p1(ii,:)))==edgeMatrix(iii,2:3)
%                     elementInside(i,ii+6)=iii;
%                     flag=0; break
%                 end
%             end
            b=find(sum(vector1(p1(ii,:)),2)==sum(edgeMatrix(:,2:3),2));
            if b~=0
                for iii=1:length(b)
                    if sort(vector1(p1(ii,:)))==edgeMatrix(b(iii),2:3)
                        elementInside(i,ii+5)=b(iii);
                        flag=0; 
                        break
                    end
                end
            end
            if flag
                edgeMatrix(num1,2:3)=sort(vector1(p1(ii,:)));
                elementInside(i,ii+5)=num1;
                num1=num1+1;               
            end
        end
    end
end
elementSurface=cat(2,elementSurface,zeros(length(elementSurface(:,1)),6));
%处理边界单元
for j=1:length(elementSurface(:,1))
    vector2=elementSurface(j,2:4);
    p2=[1 2;1 3;2 3;2 4];
    for jj=1:3
        b=find(sum(vector2(p2(jj,:)))==sum(edgeMatrix(:,2:3),2));
        for jjj=1:length(b)
            if sort(vector2(p2(jj,:)))==edgeMatrix(b(jjj),2:3)
                elementSurface(j,jj+4)=b(jjj); break
            end
        end
    end
end
    
