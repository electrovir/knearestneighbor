patterns = [0.3, 0.8; -0.3, 1.6; 0.9, 0; 1, 1];
input = [0.5, 0.2];
squares = zeros(size(patterns));
distances = zeros(1, length(patterns));

for i=1:length(patterns)
   squares(i, :) = (patterns(i, :)-input).^2;
   distance(i) = sqrt(sum(squares(i, :)));
end

%%

trainPatterns = [0.3,0.8;-0.3,1.6;0.9,0;1,1;2,2;-2,0;0,0];
trainTargetsLarge = [0, 1, 1, 0, 1, 1, 0];

testPatterns = [0.5,0.2;-1,1;0.5,1.5;2,0;0,0.5];
testTargets = [0, 1, 0, 1, 0];

results = zeros(1, length(trainPatterns));

for i = 1:length(trainPatterns)
   results(i) = sum(abs(trainPatterns(i, :) - testPatterns(5, :)));
end

results'