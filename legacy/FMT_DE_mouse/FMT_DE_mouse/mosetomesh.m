function [b_analytic] = mosetomesh(nodes,num_esource,forwardFile,forwardfym)
MeshFile           =  forwardFile ;%%%%%%%%%%moseÓÃµÄÍø¸ñ%%%%%%
fp = fopen(MeshFile);
sizenode = fscanf(fp,'%d',1);
nodes1 = fscanf(fp,'%f %f %f',[3,sizenode]);
nodes1 = nodes1';
fclose(fp);
b=forwardfym;
nodes_sur=cat(2,nodes1,b);
nodes0=zeros(length(nodes(:,1)),num_esource+3);
nodes0(:,1:3)=nodes;
for i = 1 : length(nodes0(:,1))
    node_temp1 = nodes0(i,1:3);
    d = 1000;
     for ii = 1 : length(nodes_sur(:,1))
        node_temp2 = nodes_sur(ii,1:3);
        if norm(node_temp1-node_temp2)<d
            d = norm(node_temp1-node_temp2);
            n = ii;
        end
    end
    nodes0(i,4:num_esource+3) = nodes_sur(n,4:num_esource+3);
end
b_analytic = nodes0(:,4:num_esource+3);
