function ds = processTree(dataset)

[trnX, trnY, tstX, tstY] = readData(dataset);

[model, nLeaf, trainTime, x, y1, y2] = analyzeData(trnX, trnY, tstX, tstY);

drawAnalysis(dataset, x, y1, y2, nLeaf)

drawMisclassif(dataset, model, tstX, tstY, nLeaf)

fprintf('\n===== Tree =====\n')
fprintf('dataset %d\n', dataset)
fprintf('      training time = %7.4f s\n', trainTime(1))
fprintf('         query time = %7.4f s\n', trainTime(2))
fprintf('    in-sample error = %7.4f %%\n', y1(nLeaf))
fprintf('out of sample error = %7.4f %%\n', y2(nLeaf))

ds.time1 = trainTime(1);
ds.time2 = trainTime(2);
ds.err1 = y1(nLeaf);
ds.err2 = y2(nLeaf);
end

function [modelOptimal, nOptimal, timing, minLeafSize, inSampleError, outOfSampleError] ...
    = analyzeData(trnX, trnY, tstX, tstY)
minLeafSize = 1 : 30;
inSampleError = zeros(1, length(minLeafSize));
outOfSampleError = inSampleError;
oldErr = 101;
for j = 1 : length(minLeafSize)
    [model, err, t] = train(trnX, trnY, minLeafSize(j));
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
view(modelOptimal, 'Mode', 'graph')
end

function [model, err, trainTime] = train(X, Y, minLeafSize)
tic
model = fitctree(X, Y, 'SplitCriterion', 'deviance', 'MinLeafSize', minLeafSize, ...
    'Surrogate', 'off', 'ClassNames', sort(unique(Y)));
trainTime = toc;
err = resubLoss(model) * 100;
end

function drawAnalysis(dataset, x, y1, y2, nLeaf)
close all

yMax = max(max(max(y1, y2))) + 1;
yMin = 0;
xTick = round(linspace(min(x), max(x), 7));
XTickLabel = {};
for i = 1 : 7
    XTickLabel{i} = sprintf('%d', xTick(i)); %#ok<AGROW>
end

C = lines(2);
figure('Position', [400 174.6 283.4 225.4])

plot(x, y1, '-o', 'MarkerSize', 1.5, 'MarkerFaceColor', C(1, :))
hold on
plot(x, y2, '-o', 'MarkerSize', 1.5, 'MarkerFaceColor', C(2, :))
text(x(nLeaf),y2(nLeaf),sprintf('nLeaf = %d',x(nLeaf)),'Interpreter','latex','HorizontalAlignment','center','VerticalAlignment','top','FontSize',8)
plot(x(nLeaf),y2(nLeaf),'kx','MarkerSize',6)
xlabel('min. leaf size', 'Interpreter', 'latex')
ylabel('error [%]', 'FontName', 'Times New Roman')
legend('in-sample', 'out of sample', 'FontName', 'Times New Roman','Location','best')
xlim([min(x), max(x)])
ylim([yMin, yMax])
ax = gca;
ax.FontName = 'Times New Roman';
ax.XTick = xTick;
ax.XTickLabel = XTickLabel;

filePath = fullfile('figs', sprintf('tree_error_%d', dataset));
print(filePath, '-depsc', '-painters')
end

function drawMisclassif(dataset, model, tstX, tstY, nLeaf)
predY = predict(model, tstX);
idx_correct = predY==tstY;
idx_wrong = predY~=tstY;

% classification plot
figure('Position', [700 174.6 283.4 225.4])
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

filePath = fullfile('figs', sprintf('tree_classif_%d', dataset));
print(filePath, '-depsc', '-painters')
end