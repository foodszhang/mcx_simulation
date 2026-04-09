function [ Lx ]=sorcesetup (num_esource,excite_nodes)
global nodes  elementInside 
count_nodes=length(nodes(:,1));
count_elementInside=length(elementInside(:,1));
 N=num_esource;      
% excite_nodes=importdata('E:\FMTmouseforward\intersect.txt');
distence=zeros(count_elementInside,1);
Lx=zeros(count_nodes,N);
for i=1:N;
   exsign=zeros(count_nodes,1);
   for l=1:count_elementInside
       node=elementInside(l,2:5);
       center0=nodes(node,:);
       center =0.25*sum(center0);   %%% sum是对每列求和
       distence(l) = norm(center-excite_nodes(i,:));
   end
   [va  k]=min(distence);%%%% va是返回的最小值，k是返回的最小值的位置%%%
%    min_value(i)=va ;
   c_node=elementInside(k,2:5);
   exsign(c_node)=1;
   Lx(:,i)=exsign;
end