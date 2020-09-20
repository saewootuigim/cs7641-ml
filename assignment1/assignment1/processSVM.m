function ds = processSVM(dataset)

fprintf('\n===== SVM =====\n')

kernelName = {'linear', 'rbf'};
ds = struct(kernelName{1}, [], kernelName{2}, []);

for i = 1 : length(kernelName)
    [trnX, trnY, tstX, tstY] = readData(dataset);
    
    [model, trainTime, y1, y2] = analyzeData(kernelName{i}, trnX, trnY, tstX, tstY);
    
    drawMisclassif(kernelName{i}, dataset, model, tstX, tstY)
    
    fprintf('dataset %d\n', dataset)
    fprintf('kernel %s\n', kernelName{i})
    fprintf('      training time = %7.4f s\n', trainTime(1))
    fprintf('         query time = %7.4f s\n', trainTime(2))
    fprintf('    in-sample error = %7.4f %%\n', y1)
    fprintf('out of sample error = %7.4f %%\n', y2)
    ds.(kernelName{i}).time1 = trainTime(1);
    ds.(kernelName{i}).time2 = trainTime(2);
    ds.(kernelName{i}).err1 = y1;
    ds.(kernelName{i}).err2 = y2;
end
end

function [modelOptimal, timing, inSampleError, outOfSampleError] ...
    = analyzeData(kernelName, trnX, trnY, tstX, tstY)
[model, inSampleError, t] = train(trnX, trnY, kernelName);
tic
outOfSampleError = loss(model, tstX, tstY) * 100;
queryTime = toc;
modelOptimal = model;
timing = [t, queryTime];
end

function [model, err, trainTime] = train(X, Y, kernelName)
t = templateSVM('Standardize', true, 'KernelFunction', kernelName);
tic
model = fitcecoc(X, Y, 'Learners', t, 'FitPosterior', true, ...
    'ClassNames', sort(unique(Y)));
err = resubLoss(model) * 100;
trainTime = toc;
end

function drawMisclassif(kernelName, dataset, model, tstX, tstY)
predY = predict(model, tstX);
idx_correct = predY==tstY;
idx_wrong = predY~=tstY;

% classification plot
figure('Position', [700 174.6 283.4 225.4])
plot(tstX(idx_correct, 1), tstX(idx_correct, 2), '.')
hold on
plot(tstX(idx_wrong, 1), tstX(idx_wrong, 2), 'x', 'MarkerSize', 3)
axis image

x1 = linspace(-pi, pi, 100);
y1 = -x1.^2/5+1;
x2 = cos(linspace(0, 2*pi, 100));
y2 = sin(linspace(0, 2*pi, 100)) - 0.5;
plot(x1, y1, 'k', 'LineWidth', 1)
plot(x2, y2, 'k', 'LineWidth', 1)

xlabel('$x$', 'Interpreter', 'latex')
ylabel('$y$', 'Interpreter', 'latex')

legend('correct', 'misclassified', 'Location', 'best', 'FontName', 'Times New Roman')%, 'numColumns', 2)

ax = gca;
ax.FontName = 'Times New Roman';

filePath = fullfile('figs', sprintf('SVM_classif_%s_%d', kernelName, dataset));
print(filePath, '-depsc', '-painters')
end