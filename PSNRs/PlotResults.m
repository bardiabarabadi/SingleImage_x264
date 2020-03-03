% Get data from .mat

ext='.mat';
enh=['enhanced' ext];
raw=['raw' ext];

ENH=1;
RAW=2;

to_show_data=zeros(7204,51,2);
to_show_data_mean=zeros(51,2);
to_show_data_std=zeros(51,2);


for qp=1:51
    file_name_enh = ['QP' num2str(qp) '_' enh];
    file_name_raw = ['QP' num2str(qp) '_' raw];
    
    if isfile(file_name_enh) && isfile(file_name_raw)
        load(file_name_enh)
        to_show_data(:,qp,ENH)=results_NR;
        
        found_non_zeros = find(results_NR);
        
        to_show_data_mean(qp,ENH) = mean(results_NR(found_non_zeros));
        to_show_data_std(qp,ENH) = std(results_NR(found_non_zeros));
        load(file_name_raw)
        to_show_data(:,qp,RAW)=results_NR;
        to_show_data_mean(qp,RAW) = mean(results_NR(found_non_zeros));
        to_show_data_std(qp,RAW) = std(results_NR(found_non_zeros));
        
        found_zeros = find(~to_show_data(:,qp,ENH));
        found_non_zeros = find(to_show_data(:,qp,ENH));
    end
end


to_show_data (found_zeros,:,:)=[];
to_show_data_mean(:,ENH)=inpaint_nans(to_show_data_mean(:,ENH));
to_show_data_mean(:,RAW)=inpaint_nans(to_show_data_mean(:,RAW));


close all
figure
subplot(3,2,1)
bar(to_show_data_mean(:,ENH)-to_show_data_mean(:,RAW),'b')
xlim([29,49])
title ('Mean PSNR imporvement')
xlabel ('QP')
ylabel ('PSNR_{Enhanced}-PSNR_{RAW}')

subplot(3,2,2)
bar(to_show_data_std)
legend('Enhanced', 'RAW');
xlim([29,49])
ylim([0.5 1.5])
title ('Standard Deviation PSNR')
xlabel ('QP')
ylabel ('STD')

subplot(3,2,3)
hist(to_show_data(:,30,ENH),20)
xlim([28,42])
title ('Enhanced PSNR Distribution for QP=30')
xlabel ('PSNR')

subplot(3,2,5)
hist(to_show_data(:,30,RAW),20)
xlim([28,42])
title ('RAW PSNR Distribution for QP=30')
xlabel ('PSNR')


subplot(3,2,4)
hist(to_show_data(:,46,ENH),20)
xlim([28,36])
title ('Enhanced PSNR Distribution for QP=46')
xlabel ('PSNR')

subplot(3,2,6)
hist(to_show_data(:,46,RAW),20)
xlim([28,36])
title ('RAW PSNR Distribution for QP=46')
xlabel ('PSNR')


