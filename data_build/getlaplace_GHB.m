function [Lap0,Lap1, Lap2,Lap3]=getlaplace(var)
global  nodes
point=nodes;
vertexN = size(point,1);
L1 = zeros(vertexN);
L2 = eye(vertexN);
L=L1+L2;
pp=point(I,:);
for i=1:vertexN
    r1 =0.1;
    flag_tru1 = 1;
    flag_tru2 = 1;
    while (flag_tru1 <4)
        flag_true1 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=r1);
        flag_tru1 = sum(length(flag_true1(:)));
       % if flag_tru1 <=4
            r1 = r1 + 0.01;
        %end
    end
    flag_true1 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=r1);
    flag_true2 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=1.5*r1);
    flag_true3 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=2*r1);
    for j=i+1:vertexN
        D(i,j)=(norm(pp(i,:)-pp(j,:))^2);
        D(j,i)= D(i,j);
    end
end

 DS01= exp(-D/(1*var));
 DS0=-1*DS01./repmat(sqrt(sum(DS01,1)),vertexN,1);
 Lap0=DS0+L2;
 %%
 DS02= exp(-D/(4*var));
 DS1=-1*DS02./repmat(sqrt(sum(DS02,1)),vertexN,1);
 Lap1=DS1+L2;
 %%
 DS03= exp(-D/(9*var));
 DS2=-1*DS03./repmat(sqrt(sum(DS03,1)),vertexN,1);
 Lap2=DS2+L2;
 %%
 DS04= exp(-D/(16*var));
 DS3=-1*DS04./repmat(sqrt(sum(DS04,1)),vertexN,1);
 Lap3=DS3+L2;



