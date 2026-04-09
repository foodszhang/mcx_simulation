function F = FormMatrixF(sign, fyx)
global elementInside nodes 
F  =sparse(length(nodes(:,1)),length(nodes(:,1)));
for l = 1 : length(elementInside(:,1))
    if sign(l) == 1   
    node=elementInside(l,2:5);
    %光源区域每个四面体单元中的ai,bi,ci,di
    A(2:4,:)=nodes(node,:)';A(1,:)=ones(1,4);
    D0=abs(det(A)); 
    fl=(D0)/120*ones(4);
    fl=fl+diag(diag(fl));
    fl=sparse(fl);%%%%%%   把非零元素拿出了   %%%%%
    for i = 1:4
        for j = 1:4
           F(node(i),node(j)) = F(node(i),node(j))+fl(i,j);
        end
    end
    end
end
% fy1=zeros(sizeNode);%%%% 这里可能有问题%%%%%
for j=1:length(nodes(:,1))
     F(:,j)=F(:,j)*fyx(j);
end
