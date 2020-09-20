close all

for i = 1 : 3
    % for the reproducibility
    % train set
    rng(109)
    
    nTrain = 700;
    nTest = 300;
    
    switch i
        case 1
            r = abs(randn(nTrain,1));
            th = rand(nTrain,1)*2*pi;
            x1 = 1.3*r.*cos(th);
            y1 = r.*sin(th);
        case 2
            x1 = (rand(nTrain,1)-0.5)*2*pi;
            y1 = (rand(nTrain,1)-0.5)*4;
        case 3
            x1 = linspace(-pi, pi, 28);
            y1 = linspace(-2, 2, 25);
            [x1, y1] = meshgrid(x1, y1);
            x1 = x1(:);
            y1 = y1(:);
    end
    trnX = [x1, y1];
    trnY = zeros(size(x1)) + 'a';
    trnY(-x1.^2 < 5*(y1-1)) = 'b';
    trnY(x1.^2 + (y1+0.5).^2<1) = 'c';
    trnY = toCategorical(trnY);
    
    % for the reproducibility
    % test set
    rng(109)
    
    x2 = (rand(nTest,1)-0.5)*2*pi;
    y2 = (rand(nTest,1)-0.5)*4;
    tstX = [x2, y2];
    tstY = zeros(size(x2)) + 'a';
    tstY(-x2.^2 < 5*(y2-1)) = 'b';
    tstY(x2.^2 + (y2+0.5).^2<1) = 'c';
    tstY = toCategorical(tstY);
    
    features = {'x coord','y coord'};
    label = 'class';
    
    save(sprintf('serialized%d',i),'trnX','trnY','tstX','tstY','features','label')
    X = [trnX;tstX];
    Y = [trnY;tstY];
    T = table(X(:,1),X(:,2),Y,'VariableNames',{'x coord','y coord','label'});
    writetable(T,sprintf('raw%d.csv',i))
    
    figure('Position', [300  580-(i-1)*280  600  200])
    
    x_ref_1 = linspace(-pi,pi,100);
    y_ref_1 = -x_ref_1.^2/5 + 1;
    x_ref_2 = cos(linspace(0,2*pi,100));
    y_ref_2 = sin(linspace(0,2*pi,100)) - 0.5;
    
    subplot(1,2,1)
    h = gscatter(trnX(:,1),trnX(:,2),trnY,lines(3)); h(1).MarkerSize = 5; h(2).MarkerSize = 5; h(3).MarkerSize = 5;
    legend off
    hold on
    plot(x_ref_1,y_ref_1,'k','LineWidth',1)
    plot(x_ref_2,y_ref_2,'k','LineWidth',1)
    hold off
    
    ax = gca;
    ax.FontName = 'Times New Roman';
    ax.XLabel.String = '$x$';
    ax.XLabel.Interpreter = 'latex';
    ax.YLabel.String = '$y$';
    ax.YLabel.Interpreter = 'latex';
    ax.Position(1) = 0.07;
    ax.Position(2) = 0.22;
    axis equal
    ax.XLim = [-pi, pi];
    ax.YLim = [-2, 2];
    title(sprintf('train set, n=%d',nTrain), 'FontName', 'Times New Roman', 'Units', 'normalized', 'Position', [0.5,-.43,0])
    
    subplot(1,2,2)
    h = gscatter(tstX(:,1),tstX(:,2),tstY,lines(3)); h(1).MarkerSize = 5; h(2).MarkerSize = 5; h(3).MarkerSize = 5;
    hold on
    plot(x_ref_1,y_ref_1,'k','LineWidth',1)
    plot(x_ref_2,y_ref_2,'k','LineWidth',1)
    hold off
    
    ax = gca;
    ax.FontName = 'Times New Roman';
    ax.XLabel.String = '$x$';
    ax.XLabel.Interpreter = 'latex';
    ax.YLabel.String = '$y$';
    ax.YLabel.Interpreter = 'latex';
    ax.Position(1) = 0.5;
    ax.Position(2) = 0.22;
    axis equal
    ax.XLim = [-pi, pi];
    ax.YLim = [-2, 2];
    title(sprintf('test set, n=%d',nTest), 'FontName', 'Times New Roman', 'Units', 'normalized', 'Position', [0.5,-.43,0])
    
    legend('a', 'b', 'c', 'Interpreter', 'latex', 'Units', 'normalized', 'Position', [0.9,0.45,.05,.1])
    
    fileName = fullfile('..','..','assignment1','figs',sprintf('data%d',i));
    print(fileName,'-depsc','-painters')
    print(fileName,'-dpng','-r300')
end

function catArr = toCategorical(dbl)
catArr = cell(size(dbl));
for i = 1 : length(dbl)
    catArr{i} = char(dbl(i));
end
catArr = categorical(catArr);
end