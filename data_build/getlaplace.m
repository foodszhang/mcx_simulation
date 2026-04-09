function Lap1=getlaplace
global  elementInside nodes
sizeNode=size(nodes,1);
c = zeros(sizeNode,1);
adjacency_matrix = zeros(sizeNode,sizeNode);
degree_matrix = zeros(sizeNode,sizeNode);
sigma = 1;
for i = 1:sizeNode
    element = elementInside(:,2:5); 
    [index_u, index_v] = find(element == i); 
    element_hood = unique(elementInside(index_u,2:5));
    adjacency_matrix(i,element_hood) = 1;
    adjacency_matrix(element_hood,i) = 1;
    degree_matrix(i,i) = length(element_hood) - 1;
end
adjacency_matrix = adjacency_matrix-diag(diag(adjacency_matrix));
Lap1 =eye(sizeNode,sizeNode)-degree_matrix^-0.5 * adjacency_matrix * degree_matrix^-0.5; 