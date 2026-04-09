function [] = write_dat_part(elementInside,phantom_element, phantom_point, fid, k, zone_name)

phantom_point_num = length(phantom_point);
if k == 0
    target_elements = phantom_element;
else
    target_elements = phantom_element(elementInside(:,1)==k,:);
end

if ~isempty(target_elements)
    fprintf(fid,'%s','ZONE T=');
    fprintf(fid,'%s',zone_name);
    fprintf(fid,'%s',', N=');
    fprintf(fid,'%d',phantom_point_num);
    fprintf(fid,'%s',', E=');
    fprintf(fid,'%d',length(target_elements(:,1)));
    fprintf(fid,'%s\n',', F=FEPOINT, ET=TETRAHEDRON');
    fprintf(fid,'%f\t %f\t %f\t %f\t %f\n',phantom_point(:,1:4).');   % 点坐标和能量值
    fprintf(fid,'\n \n');
    fprintf(fid,' %d\t %d\t %d\t %d\n',target_elements.');          % 单元信息
    fprintf(fid,'\n \n');
end

end