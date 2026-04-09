function surf_fym=get_surface_energy(sp3result,elementSurface,num_esource)
global  nodes
surface_node_index=unique(elementSurface(:,2:end));
surf_fym =zeros(size(nodes,1),num_esource);
surf_fym(surface_node_index,:) = sp3result(surface_node_index,:);
  