close all
clear classes
clc

for dataset = 1 : 2
    resultKNN(dataset) = processKNN(dataset); %#ok<*SAGROW>
    resultSVM(dataset) = processSVM(dataset);
    resultTree(dataset) = processTree(dataset);
    resultBoost(dataset) = processBoost(dataset);
    resultANN(dataset) = processANN(dataset);
end

%% Timing
trainTime(1,:) = [resultKNN(1).time1,resultSVM(1).rbf.time1,resultTree(1).time1,resultBoost(1).time1,resultANN(1).time1];
queryTime(1,:) = [resultKNN(1).time2,resultSVM(1).rbf.time2,resultTree(1).time2,resultBoost(1).time2,resultANN(1).time2];

trainTime(2,:) = [resultKNN(2).time1,resultSVM(2).rbf.time1,resultTree(2).time1,resultBoost(2).time1,resultANN(2).time1];
queryTime(2,:) = [resultKNN(2).time2,resultSVM(2).rbf.time2,resultTree(2).time2,resultBoost(2).time2,resultANN(2).time2];

acc(1,:) = [resultKNN(1).err2,resultSVM(1).rbf.err2,resultTree(1).err2,resultBoost(1).err2,resultANN(1).err2];
acc(2,:) = [resultKNN(2).err2,resultSVM(2).rbf.err2,resultTree(2).err2,resultBoost(2).err2,resultANN(2).err2];

figure('Position', [200  300  250  200])
bar(trainTime)
ylim([0,max(max(trainTime))]*1.2)
ylabel('train time [s]')
legend('KNN','SVM(rbf)','Tree','Boost','ANN')

ax = gca;
ax.XTickLabel = {'data set 1','data set 2'};
ax.FontName = 'Times New Roman';
ax.Legend.Position = [0.5050, 0.6139, 0.3774, 0.3500];
print('figs/timing1','-depsc','-painters')
print('figs/timing1','-dpng','-r300')

figure('Position', [450  300  250  200])
bar(queryTime)
ylim([0,max(max(queryTime))]*1.2)
ylabel('query time [s]')
legend('KNN','SVM(rbf)','Tree','Boost','ANN')

ax = gca;
ax.XTickLabel = {'data set 1','data set 2'};
ax.FontName = 'Times New Roman';
ax.Legend.Position = [0.5050, 0.6139, 0.3774, 0.3500];
print(fullfile('figs/timing2'),'-depsc','-painters')
print(fullfile('figs/timing2'),'-dpng','-r300')

figure('Position', [700  300  250  200])
bar(acc)
ylim([0,max(max(acc))]*1.2)
ylabel('out of sample error [%]')
legend('KNN','SVM(rbf)','Tree','Boost','ANN')

ax = gca;
ax.XTickLabel = {'data set 1','data set 2'};
ax.FontName = 'Times New Roman';
ax.Legend.Position = [0.5050, 0.6139, 0.3774, 0.3500];
print(fullfile('figs/error1'),'-depsc','-painters')
print(fullfile('figs/error1'),'-dpng','-r300')