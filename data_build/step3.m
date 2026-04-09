% mat to dat
clear all;
close all;

output_folder = 'results_dat';

load recon_data_9029_with_cancer.mat;

% 创建输出文件夹（如果不存在）
if ~exist(output_folder, 'dir')
    mkdir(output_folder);
end

% 获取仿体基本信息
phantom_point_num = length(nodes(:,1));
phantom_point = zeros(phantom_point_num, 4);
phantom_point(:, 1:3) = nodes(:, 1:3);  % 仿体各点坐标
phantom_element = elementInside(:, 2:5);

% 生成输出文件路径 num2str()
target_dat_name = sprintf("invivo");
target_dat_name = strrep(target_dat_name, '.', '-');
file_name = target_dat_name + '.dat';
file_path = fullfile(output_folder, file_name);

% 打开文件准备写入
fid = fopen(file_path, 'w');
if fid == -1
    warning('无法打开文件: %s', file_path);
end

% 写入文件头
fprintf(fid, 'TITLE = "Result for invivo data"\n');
fprintf(fid, 'VARIABLES = "X", "Y", "Z", "Pred Density"\n');

phantom_point(:, 4) = 0;

% 写入各器官部分
write_dat_part(elementInside, phantom_element, phantom_point, fid, 1, "体");
write_dat_part(elementInside, phantom_element, phantom_point, fid, 2, "肿瘤");

% 关闭文件
fclose(fid);
fprintf('已生成文件: %s\n', file_path);