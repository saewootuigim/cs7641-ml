function ds = processBoost(dataset)

[trnX, trnY, tstX, tstY] = readData(dataset);

[model, nLearner, trainTime, x, y1, y2] = analyzeData(trnX, trnY, tstX, tstY);

drawAnalysis(dataset, x, y1, y2, nLearner)

drawMisclassif(dataset, model, tstX, tstY, nLearner)

fprintf('\n===== Boost =====\n')
fprintf('dataset %d\n', dataset)
fprintf('      training time = %7.4f s\n', trainTime(1))
fprintf('         query time = %7.4f s\n', trainTime(2))
fprintf('    in-sample error = %7.4f %%\n', y1(nLearner))
fprintf('out of sample error = %7.4f %%\n', y2(nLearner))

ds.time1 = trainTime(1);
ds.time2 = trainTime(2);
ds.err1 = y1(nLearner);
ds.err2 = y2(nLearner);
end

function [modelOptimal, nOptimal, timing, nCycle, inSampleError, outOfSampleError] ...
    = analyzeData(trnX, trnY, tstX, tstY)
nCycle = 1 : 200;
inSampleError = zeros(1, length(nCycle));
outOfSampleError = inSampleError;
oldErr = 101;
for j = 1 : length(nCycle)
    [model, err, t] = train(trnX, trnY, nCycle(j));
    inSampleError(j) = err;
    tic
    outOfSampleError(j) = loss(model, tstX, tstY) * 100;
    queryTime = toc;
    if outOfSampleError(j) < oldErr
        modelOptimal = model;
        oldErr = outOfSampleError(j);
        nOptimal = j;
        trainOptimal = t;
        queryOptimal = queryTime;
    end
end
timing = [trainOptimal, queryOptimal];
end

function [model, err, trainTime] = train(X, Y, nCycle)
template = templateTree('MinLeafSize', 4);
tic
model = fitcensemble(X, Y, 'Method', 'AdaBoostM2', 'NumLearningCycles', nCycle, ...
    'Learners', template, 'LearnRate', 0.05, 'ClassNames', unique(Y));
trainTime = toc;
err = resubLoss(model) * 100;
end

function drawAnalysis(dataset, x, y1, y2, nLearner)
close all

yMax = max(max(max(y1, y2))) + 1;
yMin = 0;
xTick = round(linspace(min(x), max(x), 7));
XTickLabel = {};
for i = 1 : 7
    XTickLabel{i} = sprintf('%d', xTick(i)); %#ok<AGROW>
end

C = lines(2);
% error curve
figure('Position', [489  200  250  200])

plot(x, y1, '-o', 'MarkerSize', 1, 'MarkerFaceColor', C(1, :))
hold on
plot(x, y2, '-o', 'MarkerSize', 1, 'MarkerFaceColor', C(2, :))
plot(x(nLearner),y2(nLearner),'kx','MarkerSize',6)
text(x(nLearner-4),y2(nLearner),sprintf('nLearner = %d',x(nLearner)),'Interpreter','latex','HorizontalAlignment','center','VerticalAlignment','top','FontSize',8)
xlabel('number of learners', 'Interpreter', 'latex')
ylabel('error [%]', 'FontName', 'Times New Roman')
legend('in-sample', 'out of sample', 'FontName', 'Times New Roman', 'Location', 'best')
xlim([min(x), max(x)])
ylim([yMin, yMax])
ax = gca;
ax.FontName = 'Times New Roman';
ax.XTick = xTick;
ax.XTickLabel = XTickLabel;

filePath = fullfile('figs', sprintf('boost_error_%d', dataset));
print(filePath, '-depsc', '-painters')
end

function drawMisclassif(dataset, model, tstX, tstY, nLearner)
predY = predict(model, tstX);
idx_correct = predY==tstY;
idx_wrong = predY~=tstY;

% classification plot
figure('Position', [800  200  250  200])
plot(tstX(idx_correct,1),tstX(idx_correct,2),'.')
hold on
plot(tstX(idx_wrong,1),tstX(idx_wrong,2),'x','MarkerSize',3)
axis image

x1 = linspace(-pi,pi,100);
y1 = -x1.^2/5+1;
x2 = cos(linspace(0,2*pi,100));
y2 = sin(linspace(0,2*pi,100)) - 0.5;
plot(x1,y1,'k','LineWidth',1)
plot(x2,y2,'k','LineWidth',1)

xlabel('$x$','Interpreter','latex')
ylabel('$y$','Interpreter','latex')

legend('correct','misclassified','Location','best','FontName','Times New Roman')%,'numColumns',2)

ax = gca;
ax.FontName = 'Times New Roman';

filePath = fullfile('figs', sprintf('boost_classif_%d', dataset));
print(filePath, '-depsc', '-painters')
end