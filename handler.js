import * as tf from '@tensorflow/tfjs';
import runpodSdk from 'runpod-js';

const LOG = '[GPU-Worker]';

function applyAttention(sequence, windowSize, attentionUnits) {
  const attnHidden = tf.layers.dense({ units: attentionUnits, activation: 'tanh' }).apply(sequence);
  const attnScores = tf.layers.dense({ units: 1 }).apply(attnHidden);
  const attnFlat = tf.layers.reshape({ targetShape: [windowSize] }).apply(attnScores);
  const attnWeightsFlat = tf.layers.softmax().apply(attnFlat);
  const attnWeights = tf.layers.reshape({ targetShape: [windowSize, 1] }).apply(attnWeightsFlat);
  return tf.layers.multiply().apply([sequence, attnWeights]);
}

function buildModel(commodity, featureCount, windowSize, config) {
  const isEe = commodity === 'electricity';
  const l1 = config.lstmUnitsL1 || (isEe ? 256 : 128);
  const l2 = config.lstmUnitsL2 || (isEe ? 128 : 64);
  const l3 = config.lstmUnitsL3 || (isEe ? 64 : 32);
  const attnUnits = config.attentionUnits || 32;
  const outputUnits = isEe ? 96 : 1;

  const input = tf.input({ shape: [windowSize, featureCount] });
  let x = input;

  x = tf.layers.lstm({ units: l1, returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }), recurrentInitializer: 'glorotUniform' }).apply(x);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.dropout({ rate: 0.3 }).apply(x);

  x = tf.layers.lstm({ units: l2, returnSequences: true, kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }), recurrentInitializer: 'glorotUniform' }).apply(x);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.dropout({ rate: 0.2 }).apply(x);

  x = applyAttention(x, windowSize, attnUnits);

  x = tf.layers.lstm({ units: l3, returnSequences: false, kernelRegularizer: tf.regularizers.l2({ l2: 0.001 }), recurrentInitializer: 'glorotUniform' }).apply(x);
  x = tf.layers.batchNormalization().apply(x);
  x = tf.layers.dropout({ rate: 0.1 }).apply(x);

  x = tf.layers.dense({ units: l2, activation: 'relu' }).apply(x);
  const output = tf.layers.dense({ units: outputUnits, activation: 'sigmoid' }).apply(x);

  return tf.model({ inputs: input, outputs: output });
}

function serializeWeights(model) {
  const weights = model.getWeights();
  const serialized = weights.map(w => ({
    shape: w.shape,
    data: Array.from(w.dataSync()),
  }));
  weights.forEach(w => w.dispose());
  return serialized;
}

function deserializeWeights(weightData) {
  return weightData.map(w => tf.tensor(w.data, w.shape));
}

async function handler(event) {
  const startTime = Date.now();
  const input = event.input;

  const {
    code,
    commodity,
    features,
    targets,
    foundationWeights,
    config = {},
  } = input;

  if (!code || !commodity || !features || !targets) {
    return { error: 'Missing required fields: code, commodity, features, targets' };
  }

  const epochs = config.epochs || 3;
  const learningRate = config.learningRate || 0.0005;

  console.log(`${LOG} Training ${code} (${commodity}), features=[${features.length}][${features[0]?.length || 0}], targets=[${targets.length}], epochs=${epochs}`);

  try {
    const xTensor = tf.tensor3d(features);
    const windowSize = xTensor.shape[1];
    const featureCount = xTensor.shape[2];
    const yTensor = tf.tensor2d(targets);

    const model = buildModel(commodity, featureCount, windowSize, config);

    if (foundationWeights && foundationWeights.length > 0) {
      const fWeights = deserializeWeights(foundationWeights);
      const mWeights = model.getWeights();
      const toSet = [];
      const minLen = Math.min(fWeights.length, mWeights.length);
      for (let i = 0; i < mWeights.length; i++) {
        if (i < minLen && fWeights[i].shape.toString() === mWeights[i].shape.toString()) {
          toSet.push(fWeights[i].clone());
        } else {
          toSet.push(mWeights[i].clone());
        }
      }
      model.setWeights(toSet);
      fWeights.forEach(w => w.dispose());
      mWeights.forEach(w => w.dispose());
      toSet.forEach(w => w.dispose());

      for (const layer of model.layers) {
        if (layer.getClassName() === 'LSTM') {
          layer.trainable = false;
        }
      }
      console.log(`${LOG} Foundation weights applied, LSTM layers frozen`);
    }

    model.compile({
      optimizer: tf.train.adam(learningRate),
      loss: 'meanSquaredError',
      metrics: ['mae'],
    });

    const history = await model.fit(xTensor, yTensor, {
      epochs,
      batchSize: config.batchSize || 32,
      validationSplit: 0.2,
      verbose: 0,
    });

    const lastMae = history.history.val_mae
      ? history.history.val_mae[history.history.val_mae.length - 1]
      : history.history.mae[history.history.mae.length - 1];

    const trainedWeights = serializeWeights(model);

    model.dispose();
    xTensor.dispose();
    yTensor.dispose();

    const trainingTime = (Date.now() - startTime) / 1000;
    console.log(`${LOG} Done ${code}: MAE=${lastMae.toFixed(4)}, ${trainingTime.toFixed(1)}s`);

    return {
      success: true,
      weights: trainedWeights,
      mae: parseFloat(lastMae.toFixed(6)),
      epochs,
      trainingTime,
    };
  } catch (err) {
    console.error(`${LOG} Error training ${code}: ${err.message}`);
    return {
      success: false,
      error: err.message,
      trainingTime: (Date.now() - startTime) / 1000,
    };
  }
}

runpodSdk.serverless({ handler });
console.log(`${LOG} RunPod GPU worker started`);
