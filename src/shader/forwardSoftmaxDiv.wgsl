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

@group(2) @binding(0) var<storage, read_write> softmaxSumBuffer: f32;

// ソフトマックスの手順
// 1. 各要素のexp(z)を計算
// 2. exp(z)の合計を計算
// 3. 各要素を合計で割る <- このシェーダーではここ
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let threadIndex: u32 = global_id.x;
  let layerData = layerDataBuffer[layerIndex];
  if (layerData.neuronsNum <= threadIndex) {
    return;
  }

  let neuronIndex = layerData.neuronOffset + threadIndex;

  activatedValueBuffer[neuronIndex] /= softmaxSumBuffer;
}