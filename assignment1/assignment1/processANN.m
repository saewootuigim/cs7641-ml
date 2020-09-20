function ds = processANN(dataset)

[trnX, trnY, tstX, tstY] = readData(dataset);

[trnX, trnY, tstX, tstY] = modifyData(trnX, trnY, tstX, tstY);

[modelOptimal, optimalHiddenLayerSize, trainTime, x, y1, y2] = analyzeData(trnX, trnY, tstX, tstY);

drawAnalysis(dataset, x, y1, y2, optimalHiddenLayerSize)

drawMisclassif(dataset, modelOptimal, tstX, tstY)

fprintf('\n===== ANN =====\n')
fprintf('dataset %d\n', dataset)
fprintf('      training time = %7.4f s\n', trainTime(1))
fprintf('         query time = %7.4f s\n', trainTime(2))
fprintf('    in-sample error = %7.4f %%\n', y1(optimalHiddenLayerSize))
fprintf('out of sample error = %7.4f %%\n', y2(optimalHiddenLayerSize))

ds.time1 = trainTime(1);
ds.time2 = trainTime(2);
ds.err1 = y1(optimalHiddenLayerSize);
ds.err2 = y2(optimalHiddenLayerSize);
end

function [trnX, trnY, tstX, tstY] = modifyData(trnX, trnY, tstX, tstY)
trnX = trnX'; [~, n] = size(trnX);
y = trnY;
trnY = zeros(3, n);
for i = 1 : n
    if strcmp(char(y(i)), 'a')
        trnY(1,i) = 1;
    elseif strcmp(char(y(i)), 'b')
        trnY(2,i) = 1;
    else
        trnY(3,i) = 1;
    end
end
tstX = tstX'; [~, n] = size(tstX);
y = tstY;
tstY = zeros(3, n);
for i = 1 : n
    if strcmp(char(y(i)), 'a')
        tstY(1,i) = 1;
    elseif strcmp(char(y(i)), 'b')
        tstY(2,i) = 1;
    else
        tstY(3,i) = 1;
    end
end
end

function [modelOptimal, nodeOptimal, timing, hiddenLayerSize, inSampleError, outOfSampleError] ...
    = analyzeData(trnX, trnY, tstX, tstY)
hiddenLayerSize = 3 : 30;
inSampleError = zeros(1, length(hiddenLayerSize));
outOfSampleError = inSampleError;
oldErr = 101;
for j = 1 : length(hiddenLayerSize)
    [model, err, t] = train_(trnX, trnY, hiddenLayerSize(j));
    inSampleError(j) = err;
    tic
    predY  = model(tstX);
    realYind = vec2ind(tstY);
    predYind = vec2ind(predY);
    outOfSampleError(j) = sum(realYind ~= predYind)/numel(realYind) * 100;
    queryTime = toc;
    if outOfSampleError(j) < oldErr
        modelOptimal = model;
        oldErr = outOfSampleError(j);
        nodeOptimal = j;
        trainOptimal = t;
        queryOptimal = queryTime;
    end
end
timing = [trainOptimal, queryOptimal];
end

function [model, err, trainTime] = train_(X, Y, hiddenLayerSize)
tic

% Create a Network.
model = patternnet(hiddenLayerSize, 'trainscg');

% Divide the data into training, validation, and testing.
model.divideParam.trainRatio = 80/100;
model.divideParam.valRatio = 10/100;
model.divideParam.testRatio = 10/100;

% Train.
model = train(model, X, Y);
predY = model(X);
realYind = vec2ind(Y);
predYind = vec2ind(predY);
err = sum(realYind ~= predYind)/numel(realYind) * 100;
trainTime = toc;
end

function drawAnalysis(dataset, x, y1, y2, k)
% k - optimalHiddenLayerSize
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
xlabel('# nodes in hidden layer', 'FontName', 'Times New Roman')
ylabel('error [%]', 'FontName', 'Times New Roman')
legend('in-sample', 'out of sample', 'FontName', 'Times New Roman', 'Location', 'best')
xlim([min(x), max(x)])
ylim([yMin, yMax])
ax = gca;
ax.FontName = 'Times New Roman';
ax.XTick = xTick;
ax.XTickLabel = XTickLabel;

filePath = fullfile('figs', sprintf('ANN_error_%d',dataset));
print(filePath, '-depsc', '-painters')
print(filePath, '-dpng', '-r300')
end

function drawMisclassif(dataset, model, tstX, tstY)
predY = model(tstX);
% Convert 3Xn double to 1Xn double.
predY = vec2ind(predY);
tstY = vec2ind(tstY);

idx_correct = predY==tstY;
idx_wrong = predY~=tstY;
% err = sum(idx_wrong)/numel(idx_wrong)*100;

% classification plot
figure('Position', [700 174.6 283.4 225.4])
plot(tstX(1,idx_correct),tstX(2,idx_correct),'.')
hold on
plot(tstX(1,idx_wrong),tstX(2,idx_wrong),'x','MarkerSize',3)
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

filePath = fullfile('figs', sprintf('ANN_classif_%d', dataset));
print(filePath, '-depsc', '-painters')
print(filePath, '-dpng', '-r300')
end
