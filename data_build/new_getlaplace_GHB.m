function [Lap,n_Lap1, n_Lap2, n_Lap3]=new_getlaplace_GHB
global nodes 
sizeNode=size(nodes,1);
L= zeros(sizeNode);L0=L;L1=L;L2=L;L3=L;D=L;D1=L;D2=L;D3=L;
L200 = eye(sizeNode);
for i = 1:sizeNode
    r1 =1;
    flag_tru1 = 1;
    while (flag_tru1<5)
        flag_true1 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=r1);
        flag_tru1 = sum(length(flag_true1(:)));
       % if flag_tru1 <=4
            r1 = r1 + 0.01;
        %end
    end
    flag_true1 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=r1);
    flag_true2 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=2*r1);
    flag_true3 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=3*r1);
    flag_true4 = find(sum((nodes(:,1:3)-nodes(i,1:3)).^2,2).^0.5<=4*r1);

%%
%%0
I= flag_true1;
pp=nodes(I,:);
for i=1:length(I)
    for j=i+1:length(I)
        D(I(i),I(j))=(norm(pp(i,:)-pp(j,:))^2);
        D(I(j),I(i))= D(I(i),I(j));
    end
end
%%
%%2
I= flag_true2;
pp=nodes(I,:);
for i=1:length(I)
    for j=i+1:length(I)
        D1(I(i),I(j))=(norm(pp(i,:)-pp(j,:))^2);
        D1(I(j),I(i))= D1(I(i),I(j));
    end
end
 %%
 %%3
I= flag_true3;
pp=nodes(I,:);
for i=1:length(I)
    for j=i+1:length(I)
        D2(I(i),I(j))=(norm(pp(i,:)-pp(j,:))^2);
        D2(I(j),I(i))= D2(I(i),I(j));
    end
end
 %%
 %%4
I= flag_true4;
pp=nodes(I,:);
for i=1:length(I)
    for j=i+1:length(I)
    D3(I(i),I(j))=(norm(pp(i,:)-pp(j,:))^2);
    D3(I(j),I(i))= D3(I(i),I(j));
    end
end
end
% ee=sum(D)./sum(D~=0);
% L00=(repmat(1./sqrt(ee),sizeNode,1).*(-D).*repmat(1./sqrt(ee),sizeNode,1));
E0=sum(sqrt(D))/sum(D~=0);
L0= exp(E0*(-D)*E0);
L0(L0==1) = 0;
E1=sum(sqrt(D1))/sum(D1~=0);
L1= exp(E1*(-D1)*E1);
L1(L1==1) = 0;
E2=sum(sqrt(D2))/sum(D2~=0);
L2= exp(E2*(-D2)*E2);
L2(L2==1) = 0;
E3=sum(sqrt(D3))/sum(D3~=0);
L3= exp(E3*(-D3)*E3);
L3(L3==1) = 0;

D0=diag((sum(L0,2)+1).^(-1/2));
Lap=D0*(L0+L200)*D0;
D1=diag((sum(L1,2)+1).^(-1/2));
n_Lap1=D1*(L1+L200)*D1;
D2=diag((sum(L2,2)+1).^(-1/2));
n_Lap2=D2*(L2+L200)*D2;
D3=diag((sum(L3,2)+1).^(-1/2));
n_Lap3=D3*(L3+L200)*D3;




 