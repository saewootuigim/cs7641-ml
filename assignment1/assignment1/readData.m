function [trnX, trnY, tstX, tstY, label, features] = readData(dataset)
filePath = fullfile('..', 'data', 'synthetic', sprintf('serialized%d', dataset));
S = load(filePath);
trnX = S.trnX;
trnY = S.trnY;
tstX = S.tstX;
tstY = S.tstY;
label = S.label;
features = S.features;
end