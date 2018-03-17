const TEST_DATA = {
  trainPatterns: [
    [0.3, 0.8],
    [-0.3, 1.6],
    [0.9, 0],
    [1, 1]
  ],
  trainTargets: [
    0, 1, 1, 0 // 0 = A, 1 = B
  ],
  trainPatternsLarge: [
    [0.3, 0.8],
    [-0.3, 1.6],
    [0.9, 0],
    [1, 1],
    [2, 2],
    [-2, 0],
    [0, 0]
  ],
  trainTargetsLarge: [
    0, 1, 1, 0, 1, 1, 0
  ],
  testInput: [
    0.5, 0.2
  ],
  testTarget: [
    0
  ],
  testPatterns: [
    [0.5, 0.2],
    [-1, 1],
    [0.5, 1.5],
    [2, 0],
    [0, 0.5],
  ],
  testTargets: [
    0, 1, 0, 1, 0
  ]
};

const TEST_DATA_REGRESSION = {
  trainPatterns: [
    [0, 2],
    [3, 3],
    [3, 0]
  ],
  trainPatternsLarge: [
    [0, 2],
    [3, 3],
    [3, 0],
    [-2, 5],
    [-2, 0],
    [0, -3]
  ],
  trainTargets: [
    8, 3, 5
  ],
  trainTargetsLarge: [
    8, 3, 5, 10, 4, 2
  ],
  testInput: [
    1, 1
  ],
  testPatterns: [
    [1, 1],
    [4, 4],
    [10, 10],
    [0, 0]
  ],
  testTargets: [
    6.5, 2, 0, 6
  ]
};

runWebTests();

function testManhattanDistance() {
  return {
    testName: 'manhattan',
    results: [
      {
        result: NearestNeighbor.manhattanDistance(TEST_DATA.trainPatterns[0], TEST_DATA.testInput),
        target: 0.8
      },
      {
        result: NearestNeighbor.manhattanDistance(TEST_DATA.trainPatterns[1], TEST_DATA.testInput),
        target: 2.2
      },
      {
        result: NearestNeighbor.manhattanDistance(TEST_DATA.trainPatterns[2], TEST_DATA.testInput),
        target: 0.6000000000000001
      },
      {
        result: NearestNeighbor.manhattanDistance(TEST_DATA.trainPatterns[3], TEST_DATA.testInput),
        target: 1.3
      },
    ]
  };
}

function testEuclideanDistance() {
  return {
    testName: 'euclidean',
    results: [
      {
        result: NearestNeighbor.euclideanDistance(TEST_DATA.trainPatterns[0], TEST_DATA.testInput),
        target: 0.632455532033676
      },
      {
        result: NearestNeighbor.euclideanDistance(TEST_DATA.trainPatterns[1], TEST_DATA.testInput),
        target: 1.61245154965971
      },
      {
        result: NearestNeighbor.euclideanDistance(TEST_DATA.trainPatterns[2], TEST_DATA.testInput),
        target: 0.447213595499958
      },
      {
        result: NearestNeighbor.euclideanDistance(TEST_DATA.trainPatterns[3], TEST_DATA.testInput),
        target: 0.9433981132056605
      },
    ]
  };
}

function testGetNearestNeighbors() {
  return {
    testName: 'nearest', 
    results: [
      {
        result: NearestNeighbor._test_getNearestNeighbors(2, TEST_DATA.testInput, TEST_DATA.trainPatterns, TEST_DATA.trainTargets),
        target: [{distance: 0.8, output: 0}, {distance: 0.6000000000000001, output: 1}]
      }
    ]
  };
}

function testGetPatternOutput() {
  return {
    testName: 'pattern output',
    results: [
      {
        result: NearestNeighbor._test_getPatternOutput(3, TEST_DATA.testInput, TEST_DATA.trainPatterns, TEST_DATA.trainTargets, false, false),
        target: 0
      },
      {
        result: NearestNeighbor._test_getPatternOutput(3, TEST_DATA.testInput, TEST_DATA.trainPatterns, TEST_DATA.trainTargets, true, false),
        target: 1
      },
      {
        result: NearestNeighbor._test_getPatternOutput(3, TEST_DATA_REGRESSION.testInput, TEST_DATA_REGRESSION.trainPatterns, TEST_DATA_REGRESSION.trainTargets, false, true),
        target: 5.3333333333333333
      },
      {
        result: NearestNeighbor._test_getPatternOutput(3, TEST_DATA_REGRESSION.testInput, TEST_DATA_REGRESSION.trainPatterns, TEST_DATA_REGRESSION.trainTargets, true, true),
        target: 6.475409836065573
      },
    ]
  };
}

function testRun() {
  return {
    testName: 'run',
    results: [
      {
        result: NearestNeighbor.run(3, TEST_DATA_REGRESSION.trainPatternsLarge, TEST_DATA_REGRESSION.trainTargetsLarge, TEST_DATA_REGRESSION.testPatterns, TEST_DATA_REGRESSION.testTargets, false, true),
        target: {outputs: [5.333333333333333, 5.333333333333333, 6, 5.666666666666667], accuracy: 0, mse: 12.145833333333334}
      },
      {
        result: NearestNeighbor.run(3, TEST_DATA_REGRESSION.trainPatternsLarge, TEST_DATA_REGRESSION.trainTargetsLarge, TEST_DATA_REGRESSION.testPatterns, TEST_DATA_REGRESSION.testTargets, true, true),
        target: {outputs: [6.475409836065573, 3.688811188811189, 5.5903083700440535, 5.8181818181818175], accuracy: 0, mse: 8.534323357760114}
      },
      {
        result: NearestNeighbor.run(3, TEST_DATA.trainPatternsLarge, TEST_DATA.trainTargetsLarge, TEST_DATA.testPatterns, TEST_DATA.testTargets, false, false),
        target: {outputs: [0, 0, 0, 1, 0], accuracy: 0.8, mse: 0.2}
      },
      {
        result: NearestNeighbor.run(3, TEST_DATA.trainPatternsLarge, TEST_DATA.trainTargetsLarge, TEST_DATA.testPatterns, TEST_DATA.testTargets, true, false),
        target: {outputs: [0, 0, 0, 1, 0], accuracy: 0.8, mse: 0.2}
      },
      {
        result: NearestNeighbor.run(3, TEST_DATA.trainPatterns, TEST_DATA.trainTargets, [TEST_DATA.testInput], TEST_DATA.testTarget, true, false).outputs[0],
        target: 1
      },
      {
        result: NearestNeighbor.run(3, TEST_DATA.trainPatterns, TEST_DATA.trainTargets, [TEST_DATA.testInput], TEST_DATA.testTarget, false, false).outputs[0],
        target: 0
      },
    ]
  };
}

function runWebTests() {
  let results = [
    testManhattanDistance(),
    testEuclideanDistance(),
    testGetNearestNeighbors(),
    testGetPatternOutput(),
    testRun()
  ];
  
  results = results.map((result) => {
    return parseResult(result);
  });
  
  const passed = results.reduce( (total, current) => total && current.passed.reduce((a, b) => a && b, true), true);
  
  console.log('all passed:', passed, 'results:', results);
}

function parseResult(testOutput) {
  return {
    passed: compareResults(testOutput.results),
    results: testOutput.results,
    test: testOutput.testName
  };
}

function compareResults(results) {
  return results.map((testObject) => {
    return equal(testObject.result, testObject.target)  && ((testObject.condition !== undefined && testObject.condition) || testObject.condition === undefined);
  });
}

// ripped this straight out of fast-deep-equal: https://github.com/epoberezkin/fast-deep-equal
// made it browser friendly and reformated some weird lines
function equal(a, b) {
  if (a === b) return true;

  let arrA = Array.isArray(a);
  let arrB = Array.isArray(b);
  let i;

  if (arrA && arrB) {
    if (a.length != b.length) return false;
    for (i = 0; i < a.length; i++)
      if (!equal(a[i], b[i])) return false;
    return true;
  }

  if (arrA != arrB) return false;

  if (a && b && typeof a === 'object' && typeof b === 'object') {
    let keys = Object.keys(a);
    if (keys.length !== Object.keys(b).length) return false;

    let dateA = a instanceof Date;
    let dateB = b instanceof Date;
    if (dateA && dateB) return a.getTime() == b.getTime();
    if (dateA != dateB) return false;

    let regexpA = a instanceof RegExp;
    let regexpB = b instanceof RegExp;
    if (regexpA && regexpB) return a.toString() == b.toString();
    if (regexpA != regexpB) return false;

    for (i = 0; i < keys.length; i++)
      if (!Object.prototype.hasOwnProperty.call(b, keys[i])) return false;

    for (i = 0; i < keys.length; i++)
      if(!equal(a[keys[i]], b[keys[i]])) return false;

    return true;
  }

  return false;
}