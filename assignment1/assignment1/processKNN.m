function ds = processKNN(dataset)

[trnX, trnY, tstX, tstY] = readData(dataset);

[model, k, trainTime, x, y1, y2] = analyzeData(trnX, trnY, tstX, tstY);

drawAnalysis(dataset, x, y1, y2, k)

drawMisclassif(dataset, model, tstX, tstY)

fprintf('\n===== KNN =====\n')
fprintf('dataset %d\n', dataset)
fprintf('      training time = %7.4f s\n', trainTime(1))
fprintf('         query time = %7.4f s\n', trainTime(2))
fprintf('    in-sample error = %7.4f %%\n', y1(k))
fprintf('out of sample error = %7.4f %%\n', y2(k))

ds.time1 = trainTime(1);
ds.time2 = trainTime(2);
ds.err1 = y1(k);
ds.err2 = y2(k);
end

function [modelOptimal, kOptimal, timing, k, inSampleError, outOfSampleError] ...
    = analyzeData(trnX, trnY, tstX, tstY)
k = 1 : 30;
inSampleError = zeros(1, length(k));
outOfSampleError = inSampleError;
oldErr = 101;
for j = 1 : length(k)
    [model, err, t] = train(trnX, trnY, k(j));
    inSampleError(j) = err;
    tic
    outOfSampleError(j) = loss(model, tstX, tstY) * 100;
    queryTime = toc;
    if outOfSampleError(j) < oldErr
        modelOptimal = model;
        oldErr = outOfSampleError(j);
        kOptimal = j;
        trainOptimal = t;
        queryOptimal = queryTime;
    end
end
timing = [trainOptimal, queryOptimal];
end

function [model, err, trainTime] = train(X, Y, k)
tic
model = fitcknn(X, Y, 'Distance', 'seuclidean', 'NumNeighbors', k, ...
    'DistanceWeight', 'equal', 'ClassNames', sort(unique(Y)));
err = resubLoss(model) * 100;
trainTime = toc;
end

function drawAnalysis(dataset, x, y1, y2, k)
close all

yMax = max(max(max(y1, y2))) + 1;
yMin = 0;
xTick = round(linspace(min(x), max(x), 7));
XTickLabel = {};
for i = 1 : 7
    XTickLabel{i} = sprintf('%d', xTick(i)); %#ok<AGROW>
end

C = lines(2);
% error: in-sample and out of sample
figure('Position', [400 174.6 283.4 225.4])

plot(x, y1, '-o', 'MarkerSize', 1.5, 'MarkerFaceColor', C(1, :))
hold on
plot(x, y2, '-o', 'MarkerSize', 1.5, 'MarkerFaceColor', C(2, :))
plot(x(k),y2(k),'kx','MarkerSize',6)
text(x(k),y2(k),sprintf('k = %d',x(k)),'Interpreter','latex','HorizontalAlignment','center','VerticalAlignment','top','FontSize',8)
xlabel('num. neighbors', 'FontName', 'Times New Roman')
ylabel('error [%]', 'FontName', 'Times New Roman')
legend('in-sample', 'out of sample', 'FontName', 'Times New Roman', 'Location', 'best')
xlim([min(x), max(x)])
ylim([yMin, yMax])
ax = gca;
ax.FontName = 'Times New Roman';
ax.XTick = xTick;
ax.XTickLabel = XTickLabel;

filePath = fullfile('figs', sprintf('KNN_error_%d',dataset));
print(filePath, '-depsc', '-painters')
end

function drawMisclassif(dataset, model, tstX, tstY)
predY = predict(model, tstX);
idx_correct = predY==tstY;
idx_wrong = predY~=tstY;
% err = sum(idx_wrong)/numel(idx_wrong)*100;

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

filePath = fullfile('figs', sprintf('KNN_classif_%d', dataset));
print(filePath, '-depsc', '-painters')
end