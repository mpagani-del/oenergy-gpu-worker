import http from 'http';
import * as tf from '@tensorflow/tfjs';

const LOG = '[GPU-Worker]';
const weightCache = new Map();

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

function decodeBase64ToFloat32(b64) {
  const buf = Buffer.from(b64, 'base64');
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}

function decodeBase64Weights(weightsB64) {
  return weightsB64.map(w => {
    const data = decodeBase64ToFloat32(w.b64);
    return tf.tensor(Array.from(data), w.shape);
  });
}

function getFoundationWeights(input) {
  const { foundationWeightsB64, weightCacheKey, foundationWeights } = input;

  if (foundationWeightsB64 && foundationWeightsB64.length > 0) {
    const decoded = decodeBase64Weights(foundationWeightsB64);
    if (weightCacheKey) {
      weightCache.set(weightCacheKey, foundationWeightsB64);
      console.log(`${LOG} Foundation weights cached as '${weightCacheKey}' (${foundationWeightsB64.length} tensors)`);
    }
    return decoded;
  }

  if (weightCacheKey && weightCache.has(weightCacheKey)) {
    console.log(`${LOG} Using cached weights '${weightCacheKey}'`);
    return decodeBase64Weights(weightCache.get(weightCacheKey));
  }

  if (foundationWeights && foundationWeights.length > 0) {
    return foundationWeights.map(w => tf.tensor(w.data, w.shape));
  }

  return null;
}

async function handler(input) {
  const startTime = Date.now();

  const {
    code,
    commodity,
    config = {},
  } = input;

  if (!code || !commodity) {
    return { error: 'Missing required fields: code, commodity' };
  }

  let xTensor, yTensor;

  if (input.featuresB64 && input.featuresShape) {
    const featData = decodeBase64ToFloat32(input.featuresB64);
    xTensor = tf.tensor(Array.from(featData), input.featuresShape);
  } else if (input.features) {
    xTensor = tf.tensor3d(input.features);
  } else {
    return { error: 'Missing features data' };
  }

  if (input.targetsB64 && input.targetsShape) {
    const targData = decodeBase64ToFloat32(input.targetsB64);
    yTensor = tf.tensor(Array.from(targData), input.targetsShape);
  } else if (input.targets) {
    yTensor = tf.tensor2d(input.targets);
  } else {
    xTensor.dispose();
    return { error: 'Missing targets data' };
  }

  const windowSize = xTensor.shape[1];
  const featureCount = xTensor.shape[2];
  const epochs = config.epochs || 3;
  const learningRate = config.learningRate || 0.0005;

  console.log(`${LOG} Training ${code} (${commodity}), shape=[${xTensor.shape}], targets=[${yTensor.shape}], epochs=${epochs}`);

  try {
    const model = buildModel(commodity, featureCount, windowSize, config);

    const fWeights = getFoundationWeights(input);
    if (fWeights && fWeights.length > 0) {
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
    xTensor.dispose();
    yTensor.dispose();
    return {
      success: false,
      error: err.message,
      trainingTime: (Date.now() - startTime) / 1000,
    };
  }
}

const server = http.createServer(async (req, res) => {
  if (req.method === 'POST') {
    const chunks = [];
    req.on('data', chunk => { chunks.push(chunk); });
    req.on('end', async () => {
      try {
        const body = Buffer.concat(chunks).toString();
        const parsed = JSON.parse(body);
        const input = parsed.input || parsed;
        const result = await handler(input);
        res.writeHead(200, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ output: result, status: 'COMPLETED' }));
      } catch (err) {
        res.writeHead(500, { 'Content-Type': 'application/json' });
        res.end(JSON.stringify({ error: err.message, status: 'FAILED' }));
      }
    });
  } else if (req.method === 'GET') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'healthy', weightCacheKeys: Array.from(weightCache.keys()) }));
  } else {
    res.writeHead(405);
    res.end();
  }
});

const PORT = process.env.PORT || 8000;
server.listen(PORT, () => {
  console.log(`${LOG} HTTP server listening on port ${PORT}`);
});
