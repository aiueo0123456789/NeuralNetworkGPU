struct LayerData {
  neuronOffset: u32,
  connectionOffset: u32,
  neuronsNum: u32,
  activationFunction: u32
}

@group(0) @binding(0) var<storage, read_write> valueBuffer: array<f32>; // 活性化関数適応前
@group(0) @binding(1) var<storage, read_write> activatedValueBuffer: array<f32>; // 活性化関数適応後
@group(0) @binding(2) var<storage, read> weightBuffer: array<f32>; // 重み
@group(0) @binding(3) var<storage, read> biasBuffer: array<f32>; // バイアス
@group(0) @binding(4) var<storage, read> layerDataBuffer: array<LayerData>; // レイヤー情報

@group(1) @binding(0) var<uniform> layerIndex: u32;

fn Sigmoid(weightedInput: f32) -> f32 {
  return 1.0 / (1.0 + exp(-weightedInput));
}

fn ReLU(weightedInput: f32) -> f32 {
  return max(weightedInput, 0.0);
}

fn LeakyReLU(weightedInput: f32) -> f32 {
  return select(0.01 * weightedInput, weightedInput, weightedInput > 0.0);
}

// ソフトマックスはここでは行わない
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let threadIndex: u32 = global_id.x;
  let layerData = layerDataBuffer[layerIndex];
  if (layerData.neuronsNum <= threadIndex) {
    return;
  }

  let prevLayerData = layerDataBuffer[layerIndex - 1u];

  let neuronIndex = layerData.neuronOffset + threadIndex;

  let inputsNum: u32 = prevLayerData.neuronsNum;

  var weightedInput: f32 = biasBuffer[neuronIndex];
  for (var inIndex: u32 = 0u; inIndex < inputsNum; inIndex ++) {
    let weightIndex = layerData.connectionOffset + threadIndex * inputsNum + inIndex;
    weightedInput += activatedValueBuffer[prevLayerData.neuronOffset + inIndex] * weightBuffer[weightIndex];
  }
  valueBuffer[neuronIndex] = weightedInput;
  let activationFn = layerData.activationFunction;
  if (activationFn == 1u) {
    activatedValueBuffer[neuronIndex] = Sigmoid(weightedInput);
  } else if (activationFn == 2u) {
    activatedValueBuffer[neuronIndex] = ReLU(weightedInput);
  } else if (activationFn == 3u) {
    activatedValueBuffer[neuronIndex] = LeakyReLU(weightedInput);
  } else if (activationFn == 4u) {
    activatedValueBuffer[neuronIndex] = weightedInput;
  } else {
    activatedValueBuffer[neuronIndex] = weightedInput;
  }
}