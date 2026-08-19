struct LayerData {
  neuronOffset: u32,
  connectionOffset: u32,
  neuronsNum: u32,
  activationFunction: u32
}

@group(0) @binding(0) var<storage, read> valueBuffer: array<f32>; // 活性化関数適応前
@group(0) @binding(1) var<storage, read> activatedValueBuffer: array<f32>; // 活性化関数適応後
@group(0) @binding(2) var<storage, read> weightBuffer: array<f32>; // 重み
@group(0) @binding(3) var<storage, read_write> biasBuffer: array<f32>; // バイアス
@group(0) @binding(4) var<storage, read_write> nodeValueBuffer: array<f32>; // ノードバリュー
@group(0) @binding(5) var<storage, read_write> newWeightBuffer: array<f32>; // 重み
@group(0) @binding(6) var<storage, read> layerDataBuffer: array<LayerData>; // レイヤー情報

@group(1) @binding(0) var<uniform> layerIndex: u32;

@group(2) @binding(0) var<storage, read> targetBuffer: array<f32>; // 目標
@group(2) @binding(1) var<uniform> learningRate: f32; // 学習率

fn SigmoidDerivative(weightedInput: f32) -> f32 {
  let a = 1.0 / (1.0 + exp(-weightedInput));
  return a * (1.0 - a);
}

fn ReLUDerivative(x: f32) -> f32 {
  return select(0.0, 1.0, x > 0.0);
}

fn LeakyReLUDerivative(x: f32) -> f32 {
  return select(0.01, 1.0, x > 0.0);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let threadIndex: u32 = global_id.x;
  let layerData = layerDataBuffer[layerIndex];
  if (layerData.neuronsNum <= threadIndex) {
    return;
  }

  let nextLayerData = layerDataBuffer[layerIndex + 1u];
  let prevLayerData = layerDataBuffer[layerIndex - 1u];

  let neuronsNum = layerData.neuronsNum;
  let nextNeuronsNum = nextLayerData.neuronsNum;
  let neuronIndex = layerData.neuronOffset + threadIndex;

  var nodeValue: f32 = 0.0;
  for (var i: u32 = 0u; i < nextNeuronsNum; i ++) {
    let weightIndex = nextLayerData.connectionOffset + i * neuronsNum + threadIndex;
    nodeValue += weightBuffer[weightIndex] * nodeValueBuffer[nextLayerData.neuronOffset + i];
  }

  var value: f32 = valueBuffer[neuronIndex];
  let activationFn = layerData.activationFunction;

  if (activationFn == 1u) {
    nodeValue *= SigmoidDerivative(value);
  } else if (activationFn == 2u) {
    nodeValue *= ReLUDerivative(value);
  } else if (activationFn == 3u) {
    nodeValue *= LeakyReLUDerivative(value);
  }

  let inputNum: u32 = prevLayerData.neuronsNum;
  for (var inIndex: u32 = 0u; inIndex < inputNum; inIndex ++) {
    let weightIndex = layerData.connectionOffset + threadIndex * inputNum + inIndex;
    let derivativeCostWrtWeight: f32 = activatedValueBuffer[prevLayerData.neuronOffset + inIndex] * nodeValue;
    newWeightBuffer[weightIndex] = weightBuffer[weightIndex] - derivativeCostWrtWeight * learningRate;
  }
  biasBuffer[neuronIndex] -= nodeValue * learningRate;
  nodeValueBuffer[neuronIndex] = nodeValue;
}