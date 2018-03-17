const NearestNeighbor = (() => {
  
  function manhattanDistance(a, b) {
    return a.reduce((sum, aValue, index) => {
      return sum + Math.abs(aValue - b[index]);
    }, 0);
  }
  
  function euclideanDistance(a, b) {
    return Math.sqrt(a.reduce((sum, aValue, index) => {
      return sum + Math.pow((aValue - b[index]), 2);
    }, 0));
  }
  
  function getNearestNeighbors(k, pattern, trainPatterns, trainOutputs, distanceFunction = manhattanDistance) {
    let nearest = [];
    
    return trainPatterns.reduce((nearest, trainPattern, trainIndex) => {
      const distance = distanceFunction(trainPattern, pattern);
      
      if (nearest.length < k) {
        nearest.push({
          distance: distance,
          output: trainOutputs[trainIndex]
        });
        
        nearest.sort((a, b) => {
          return b.distance - a.distance;
        });
      }
      else if (distance < nearest[0].distance) {
        nearest[0] = {
          distance: distance,
          output: trainOutputs[trainIndex]
        };
        
        nearest.sort((a, b) => {
          return b.distance - a.distance;
        });
      }
      
      return nearest;
    }, nearest);
  }
  
  function getPatternOutput(k, pattern, trainPatterns, trainOutputs, isDistanceWeighted, isRegression, distanceFunction) {
    let nearest = getNearestNeighbors(k, pattern, trainPatterns, trainOutputs, distanceFunction);
    
    // Normal, non-regression algorithm
    if (!isRegression) {
      return nearest.reduce((accum, current) => {
        if (!accum.votes.hasOwnProperty(current.output)) {
          accum.votes[current.output] = 0;
        }
        
        // distance weighting
        if (isDistanceWeighted) {
          accum.votes[current.output] += 1/Math.pow(current.distance, 2);
        }
        else {
          accum.votes[current.output]++;
        }
        
        if (accum.votes[current.output] > accum.mode.count) {
          accum.mode.count = accum.votes[current.output];
          accum.mode.value = current.output;
        }
        
        return accum;
      }, {votes: [], mode: {count: 0, value: -1}}).mode.value;
    }
    
    // regression algorithm
    else if (!isDistanceWeighted) {
      return nearest.reduce((sum, neighbor) => {
        return sum + neighbor.output;
      }, 0) / nearest.length;
    }
    
    // regression algorithm with distance weighting
    else {
      
      let top = nearest.reduce((sum, neighbor) => {
        return sum + neighbor.output / Math.pow(neighbor.distance, 2);
      }, 0);
      let bottom = nearest.reduce((sum, neighbor) => {
        return sum + 1 / Math.pow(neighbor.distance, 2);
      }, 0);
      
      return top / bottom;
    }
  }
  
  function run(k, trainPatterns, trainTargets, testPatterns, testTargets, isDistanceWeighted, isRegression, distanceFunction = manhattanDistance) {
    const outputs = testPatterns.map(pattern => getPatternOutput(k, pattern, trainPatterns, trainTargets, isDistanceWeighted, isRegression, distanceFunction));
    
    const accuracy = outputs.reduce((sum, output, index) => {
      return sum + (output === testTargets[index]);
    }, 0) / outputs.length;
    
    const meanSquaredError = outputs.reduce((sum, output, index) => {
      return sum + Math.pow(testTargets[index] - output, 2);
    }, 0) / outputs.length;
    
    return {
      outputs: outputs,
      accuracy: accuracy,
      mse: meanSquaredError
    };
  }
  
  return {
    _test_getNearestNeighbors: getNearestNeighbors,
    _test_getPatternOutput: getPatternOutput,
    
    manhattanDistance: manhattanDistance,
    euclideanDistance: euclideanDistance,
    run: run
  };
})();

// set module exports if in node
if (typeof module !== 'undefined' && typeof module === 'object') {
  Object.assign(module.exports, NearestNeighbor);
}