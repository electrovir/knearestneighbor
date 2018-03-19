const ARFF = require('arff-toolkit');
const KNN = require('./nearestNeighbor.js');
const FS = require('fs');
const CP = require('child_process');

function saveData(results, outputFile, regression, weighted, normalize, startTime) {
  let timeText = '';
  let difference = (Number(new Date()) - startTime);
  
  if (difference < 1000) {
    timeText = difference + ' microseconds';
  }
  else if (difference / 1000 > 30) {
    timeText = ((difference / 1000) / 60) + ' minutes';
  }
  else {
    timeText = (difference / 1000) + ' seconds';
  }
  
  const outputString = 'regression: ' + regression + '\nweighted: ' + weighted + '\nnormalize: ' + normalize + '\ntotal time: ' + timeText + '\n' + JSON.stringify(results, null, 2);
  
  if (outputFile) {
    FS.writeFileSync(outputFile, outputString + '\nstartTime: ' + startTime + '\nstopTime: ' + Number(new Date()));
  }
  else {
    console.log(outputString);
  }
}

function fixUnknowns(data) {
  data.data.forEach((entry) => {
    for (let feature in entry) {
      if (entry.hasOwnProperty(feature) && isNaN(entry[feature])) {
        entry[feature] = 0.5;
      }
    }
  });
}

function runTest(startTime, trainFile, testFile, k, regression, weighted, normalize, outputFile, multithreaded) {
  console.log('starting on ' + outputFile);
  
  ARFF.loadArff(trainFile, (trainData) => {
    ARFF.loadArff(testFile, (testData) => {
      if (normalize) {
        trainData.normalize();
        testData.normalize();
        fixUnknowns(trainData);
        fixUnknowns(testData);
      }
      const train = ARFF.arffToInputs(trainData);
      const test = ARFF.arffToInputs(testData);
      
      if (!Array.isArray(k)) {
        if (typeof k !== 'number') {
          throw new Error('Invalid k value!');
        }
        k = [k];
      }
      
      let results = [];
      
      const total = k.length;
      
      k.forEach((kValue, kIndex) => {
        if (multithreaded) {
          const childData = {
            kValue: kValue,
            trainPatterns: train.patterns,
            trainTargets: train.targetColumns[0],
            testPatterns: test.patterns,
            testTargets: test.targetColumns[0],
            weighted: weighted,
            regression: regression
          };
          
          const child = CP.fork('./nearestNeighbor.js');
          
          console.log('starting ' + outputFile + ' child ' + kIndex);
          
          child.send(childData);
          child.on('message', (result) => {
            result.k = kValue;
            delete result.outputs;
            
            results.push(result);
            
            if (results.length === total) {
              console.log('finishing ' + outputFile);
              saveData(results, outputFile + '_parallel.txt', regression, weighted, normalize, startTime);
            }
          });
        }
        else {
          result = KNN.run(kValue, train.patterns, train.targetColumns[0], test.patterns, test.targetColumns[0], weighted, regression);
          
          result.k = kValue;
          
          delete result.outputs;
          results.push(result);
        }
      
      });
      
      if (!multithreaded) {
        saveData(results, outputFile + '.txt', regression, weighted, normalize, startTime);
      }
      
    });
  });
}

module.exports = {
  runTest: runTest
};