function X=given(elementInside,nodes)
  elementinside1=elementInside;
  fluo_flag=find(elementinside1(:,1)==7);%%%%%寻找荧光团的位置
  node_count=length(nodes(:,1));
  sign1=zeros( node_count,1);
  for i=1:length(fluo_flag(:,1))
      element=fluo_flag(i);
      flag=elementinside1(element,2:5);
      sign1(flag)=1;
  end
  X=0.05*sign1;
 