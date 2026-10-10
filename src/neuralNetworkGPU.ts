
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  throw new Error("WebGPU adapterを取得できませんでした");
}

const device = await adapter.requestDevice({
  requiredLimits: {
    maxStorageBuffersPerShaderStage: 10,
  },
});

const forwardBindGroupLayout0 = device.createBindGroupLayout({
  label: "forward-bind-group-layout-0",
  entries: [
    { // binding(0) valueBuffer : storage, read_write
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    { // binding(1) activatedValueBuffer : storage, read_write
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    { // binding(2) weightBuffer : storage, read
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    },
    { // binding(3) biasBuffer : storage, read
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    },
    { // binding(4) layerDataBuffer : storage, read
      binding: 4,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    },
  ],
});

const bindGroupLayout1 = device.createBindGroupLayout({
  label: "forward-bind-group-layout-1",
  entries: [
    { // binding(0) layerIndex : uniform
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" },
    },
  ],
});

const forwardSoftmaxBindGroupLayout = device.createBindGroupLayout({
  label: "forward-softmax-bind-group-layout",
  entries: [
    { // binding(0) softmaxSumBuffer : storage, read_write
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
  ],
});

const forwardPipelineLayout = device.createPipelineLayout({
  label: "forward-pipeline-layout",
  bindGroupLayouts: [forwardBindGroupLayout0, bindGroupLayout1],
});

const forwardSoftmaxPipelineLayout = device.createPipelineLayout({
  label: "forward-pipeline-layout",
  bindGroupLayouts: [forwardBindGroupLayout0, bindGroupLayout1, forwardSoftmaxBindGroupLayout],
});

const backwardBindGroupLayout0 = device.createBindGroupLayout({
  label: "backward-bind-group-layout-0",
  entries: [
    { // binding(0) valueBuffer : storage, read
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    },
    { // binding(1) activatedValueBuffer : storage, read
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    },
    { // binding(2) weightBuffer : storage, read
      binding: 2,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    },
    { // binding(3) biasBuffer : storage, read_write
      binding: 3,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    { // binding(4) nodeValueBuffer : storage, read_write
      binding: 4,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
     { // binding(5) newWeightBuffer : storage, read
      binding: 5,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "storage" },
    },
    { // binding(6) layerDataBuffer : storage, read
      binding: 6,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    },
  ],
});

const backwardBindGroupLayout2 = device.createBindGroupLayout({
  label: "backward-bind-group-layout-2",
  entries: [
    { // binding(0) targetBuffer : storage, read
      binding: 0,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "read-only-storage" },
    },
    { // binding(1) learningRate : uniform
      binding: 1,
      visibility: GPUShaderStage.COMPUTE,
      buffer: { type: "uniform" },
    },
  ],
});

const backwardPipelineLayout = device.createPipelineLayout({
  label: "backward-pipeline-layout",
  bindGroupLayouts: [
    backwardBindGroupLayout0,
    bindGroupLayout1,
    backwardBindGroupLayout2,
  ],
});

const forwardPipeline = device.createComputePipeline({
  label: "forwardPipeline",
  layout: forwardPipelineLayout,
  compute: {
    module: device.createShaderModule({
      label: "forwardShaderModel",
      code: `
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
}`
    }),
    entryPoint: "main",
  },
});

const forwardSoftmaxSumPipeline = device.createComputePipeline({
  label: "forwardSoftmaxSumPipeline",
  layout: forwardSoftmaxPipelineLayout,
  compute: {
    module: device.createShaderModule({
      label: "forwardSoftmaxSumShaderModel",
      code: `
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
// 1. 各要素のexp(z)を計算 <- このシェーダーではここ
// 2. exp(z)の合計を計算 <- このシェーダーではここ
// 3. 各要素を合計で割る
@compute @workgroup_size(1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
  let layerData = layerDataBuffer[layerIndex];

  var max = activatedValueBuffer[layerData.neuronOffset];
  for (var i = 1u; i < layerData.neuronsNum; i ++) {
    let neuronIndex = layerData.neuronOffset + i;
    let value = activatedValueBuffer[neuronIndex];
    if (max < value) {
      max = value;
    }
  }

  for (var i = 0u; i < layerData.neuronsNum; i ++) {
    let neuronIndex = layerData.neuronOffset + i;
    activatedValueBuffer[neuronIndex] = exp(activatedValueBuffer[neuronIndex] - max);
  }

  var sum = 0.0;
  for (var i = 0u; i < layerData.neuronsNum; i ++) {
    let neuronIndex = layerData.neuronOffset + i;
    sum += activatedValueBuffer[neuronIndex];
  }
  softmaxSumBuffer = sum;
}`
    }),
    entryPoint: "main",
  },
});

const forwardSoftmaxDivPipeline = device.createComputePipeline({
  label: "forwardSoftmaxDivPipeline",
  layout: forwardSoftmaxPipelineLayout,
  compute: {
    module: device.createShaderModule({
      label: "forwardSoftmaxDivShaderModel",
      code: `
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
}`
    }),
    entryPoint: "main",
  },
});

const backwardOutputPipeline = device.createComputePipeline({
  label: "backwardOutputPipeline",
  layout: backwardPipelineLayout,
  compute: {
    module: device.createShaderModule({
      label: "backwardOutputShaderModel",
      code: `
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

  let neuronIndex = layerData.neuronOffset + threadIndex;

  let activationFn = layerData.activationFunction;
  var nodeValue: f32 = 0.0;
  if (activationFn == 4u) { // softmaxとクロスエントロピーの微分はこうなるらしい理由は知らん
    nodeValue = activatedValueBuffer[neuronIndex] - targetBuffer[threadIndex];
  } else {
    nodeValue = 2.0 * (activatedValueBuffer[neuronIndex] - targetBuffer[threadIndex]);

    var value: f32 = valueBuffer[neuronIndex];

    if (activationFn == 1u) {
      nodeValue *= SigmoidDerivative(value);
    } else if (activationFn == 2u) {
      nodeValue *= ReLUDerivative(value);
    } else if (activationFn == 3u) {
      nodeValue *= LeakyReLUDerivative(value);
    }
  }

  let prevLayerData = layerDataBuffer[layerIndex - 1u];

  let inputNum: u32 = prevLayerData.neuronsNum;
  for (var inIndex: u32 = 0u; inIndex < inputNum; inIndex ++) {
    let weightIndex = layerData.connectionOffset + threadIndex * inputNum + inIndex;
    let derivativeCostWrtWeight: f32 = activatedValueBuffer[prevLayerData.neuronOffset + inIndex] * nodeValue;
    newWeightBuffer[weightIndex] = weightBuffer[weightIndex] - derivativeCostWrtWeight * learningRate;
  }
  biasBuffer[neuronIndex] -= nodeValue * learningRate;
  nodeValueBuffer[neuronIndex] = nodeValue;
}`
    }),
    entryPoint: "main",
  },
});
const backwardHiddenPipeline = device.createComputePipeline({
  label: "backwardHiddenPipeline",
  layout: backwardPipelineLayout,
  compute: {
    module: device.createShaderModule({
      label: "backwardHiddenShaderModel",
      code: `
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
}`
    }),
    entryPoint: "main",
  },
});

const ActivationFunction = {
  None: 0,
  Sigmoid: 1,
  ReLU: 2,
  LeakyReLU: 3,
  Softmax: 4,
} as const;
type ActivationFunction = typeof ActivationFunction[keyof typeof ActivationFunction];


interface LayerData {
  neuronsNum: number;
  activationFunction: ActivationFunction;
}

export interface NeuralNetworkGPUData {
  inputLayer: LayerData,
  hiddenLayers: LayerData[],
  outputLayer: LayerData,
}

class GPULayerData {
  public neuronOffset: number;
  public connectionOffset: number;
  public neuronsNum: number;
  public activationFunction: number;
  constructor(
    neuronOffset: number,
    connectionOffset: number,
    neuronsNum: number,
    activationFunction: number
  ) {
    this.neuronOffset = neuronOffset;
    this.connectionOffset = connectionOffset;
    this.neuronsNum = neuronsNum;
    this.activationFunction = activationFunction;
  }
}

export class NeuralNetworkGPU {
  static ActivationFunction = ActivationFunction;

  public layersNum: number;
  public layerDatas: GPULayerData[];
  private learningRateBuffer: GPUBuffer; // 学習時の学習率
  private targetBuffer: GPUBuffer; // 学習時の目標値
  private nodeValueBuffer: GPUBuffer; // 学習時の伝播用
  private softmaxSumBuffer: GPUBuffer; // ソフトマックス用の合計値
  private newWeightBuffer: GPUBuffer; // 学習時の新しい重み
  private valueBuffer: GPUBuffer; // 活性化関数適応前（学習用に保持する）: /ニューロン
  private activatedValueBuffer: GPUBuffer; // 活性化関数適応後: /ニューロン
  private weightBuffer: GPUBuffer; // 重み: /前の層のニューロンxニューロン
  private biasBuffer: GPUBuffer; // バイアス /ニューロン
  private layerDataBuffer: GPUBuffer; // オフセット・活性化関数の種類: /層
  private layerIndexBuffers: GPUBuffer[]; // レイヤーindex
  private bindGroup1s: GPUBindGroup[]; // レイヤーindex
  private forwardSoftmaxBindGroup: GPUBindGroup;
  private forwardBindGroup0: GPUBindGroup;
  private backwardBindGroup0: GPUBindGroup;
  private backwardBindGroup2: GPUBindGroup;

  constructor(data: NeuralNetworkGPUData) {
    this.layersNum = 2 + data.hiddenLayers.length;
    const layers = [data.inputLayer, ...data.hiddenLayers, data.outputLayer];

    this.layerDatas = [];
    let neuronOffset = 0;
    let connectionOffset = 0;
    for (let li = 0; li < layers.length; li ++) {
      const layer = layers[li];
      const layerNeuronsNum = layer.neuronsNum;
      const prevLayerNeuronsNum = layers[li - 1]?.neuronsNum ?? 0;
      this.layerDatas.push(new GPULayerData(neuronOffset, connectionOffset, layerNeuronsNum, layer.activationFunction));
      neuronOffset += layerNeuronsNum;
      connectionOffset += layerNeuronsNum * prevLayerNeuronsNum;
    }
    const neuronsNum = neuronOffset;
    const connectionsNum = connectionOffset;
    console.log(this.layerDatas)

    this.softmaxSumBuffer = device.createBuffer({
      label: "softmaxSumBuffer",
      size: 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });

    this.layerIndexBuffers = layers.map((layer, li) => {
      const buffer = device.createBuffer({
        label: `layerIndexBuffer-${li}`,
        size: 4,
        usage: GPUBufferUsage.UNIFORM,
        mappedAtCreation: true,
      });
      new Uint32Array(buffer.getMappedRange()).set([li]);
      buffer.unmap();
      return buffer;
    });

    this.forwardSoftmaxBindGroup = device.createBindGroup({
      label: "forwardSoftmaxBindGroup",
      layout: forwardSoftmaxBindGroupLayout,
      entries: [{
        binding: 0,
        resource: {
          buffer: this.softmaxSumBuffer,
        },
      }]
    });

    this.bindGroup1s = this.layerIndexBuffers.map(buffer => device.createBindGroup({
      label: "bindGroup1",
      layout: bindGroupLayout1,
      entries: [{
        binding: 0,
        resource: {
          buffer: buffer,
        },
      }]
    }));

    this.layerDataBuffer = device.createBuffer({
      label: "layerDataBuffer",
      size: this.layersNum * (4) * 4,
      usage: GPUBufferUsage.STORAGE,
      mappedAtCreation: true,
    });
    new Uint32Array(this.layerDataBuffer.getMappedRange()).set(this.layerDatas.map(layerData => [layerData.neuronOffset, layerData.connectionOffset, layerData.neuronsNum, layerData.activationFunction]).flat());
    this.layerDataBuffer.unmap();

    this.learningRateBuffer = device.createBuffer({
      label: "learningRateBuffer",
      size: 4,
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    this.targetBuffer = device.createBuffer({
      label: "targetBuffer",
      size: data.outputLayer.neuronsNum * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.nodeValueBuffer = device.createBuffer({
      label: "nodeValueBuffer",
      size: neuronsNum * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.newWeightBuffer = device.createBuffer({
      label: "newWeightBuffer",
      size: connectionsNum * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });

    this.valueBuffer = device.createBuffer({
      label: "valueBuffer",
      size: neuronsNum * 4,
      usage: GPUBufferUsage.STORAGE,
    });
    this.activatedValueBuffer = device.createBuffer({
      label: "activatedValueBuffer",
      size: neuronsNum * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
    });
    this.biasBuffer = device.createBuffer({
      label: "biasBuffer",
      size: neuronsNum * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });
    this.weightBuffer = device.createBuffer({
      label: "weightBuffer",
      size: connectionsNum * 4,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
    });

    this.forwardBindGroup0 = device.createBindGroup({
      layout: forwardBindGroupLayout0,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.valueBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.activatedValueBuffer,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.weightBuffer,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.biasBuffer,
          },
        },
        {
          binding: 4,
          resource: {
            buffer: this.layerDataBuffer,
          },
        },
      ]
    })

    this.backwardBindGroup0 = device.createBindGroup({
      layout: backwardBindGroupLayout0,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.valueBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.activatedValueBuffer,
          },
        },
        {
          binding: 2,
          resource: {
            buffer: this.weightBuffer,
          },
        },
        {
          binding: 3,
          resource: {
            buffer: this.biasBuffer,
          },
        },
        {
          binding: 4,
          resource: {
            buffer: this.nodeValueBuffer,
          },
        },
        {
          binding: 5,
          resource: {
            buffer: this.newWeightBuffer,
          },
        },
        {
          binding: 6,
          resource: {
            buffer: this.layerDataBuffer,
          },
        },
      ]
    })
    this.backwardBindGroup2 = device.createBindGroup({
      layout: backwardBindGroupLayout2,
      entries: [
        {
          binding: 0,
          resource: {
            buffer: this.targetBuffer,
          },
        },
        {
          binding: 1,
          resource: {
            buffer: this.learningRateBuffer,
          },
        },
      ]
    })
  }

  public setLearningRateBuffer(learningRate: number): void {
    device.queue.writeBuffer(this.learningRateBuffer, 0, new Float32Array([learningRate]));
  }

  public setRandom(range: number = 1): void {
    device.queue.writeBuffer(this.biasBuffer, 0, new Float32Array(this.biasBuffer.size / 4).map(() => Math.random() * range * 2 - range));
    device.queue.writeBuffer(this.weightBuffer, 0, new Float32Array(this.weightBuffer.size / 4).map(() => Math.random() * range * 2 - range));
  }

  public async getAnswer(): Promise<Float32Array> {
    const outputLayerData = this.layerDatas[this.layerDatas.length - 1]
    const count = outputLayerData.neuronsNum;
    const offset = outputLayerData.neuronOffset;
    // const count = outputLayerData.neuronsNum + outputLayerData.neuronOffset;
    // const offset = 0;
    const readBuffer = device.createBuffer({
      size: 4 * count,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();

    encoder.copyBufferToBuffer(this.activatedValueBuffer, 4 * offset, readBuffer, 0, 4 * count);

    device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);

    const values = new Float32Array(readBuffer.getMappedRange().slice());

    readBuffer.unmap();
    readBuffer.destroy();
    return values;
  }

  public async softmaxSum(): Promise<Float32Array> {
    const readBuffer = device.createBuffer({
      size: 4,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    const encoder = device.createCommandEncoder();

    encoder.copyBufferToBuffer(this.softmaxSumBuffer, 0, readBuffer, 0, 4);

    device.queue.submit([encoder.finish()]);

    await readBuffer.mapAsync(GPUMapMode.READ);

    const values = new Float32Array(readBuffer.getMappedRange().slice());

    readBuffer.unmap();
    readBuffer.destroy();
    return values;
  }

  public async getCost(input: GPUAllowSharedBufferSource, target: Float32Array): Promise<number> {
    this.run(input);
    const ans = await this.getAnswer();
    let cost = 0;
    if (this.layerDatas[this.layerDatas.length - 1].activationFunction === ActivationFunction.Softmax) { // クロスエントロピー誤差
      const delta = 1e-7; // log(0) でInfinityにならないように
      for (let i = 0; i < target.length; i++) {
        cost -= Math.log(ans[i] + delta) * target[i];
      }
    } else { // 2乗誤差
      for (let i = 0; i < target.length; i++) {
        cost += (ans[i] - target[i]) ** 2;
      }
    }
    return cost;
  }

  public run(input: GPUAllowSharedBufferSource): void {
    device.queue.writeBuffer(this.activatedValueBuffer, 0, input);
    const computeCommandEncoder = device.createCommandEncoder();
    const computePassEncoder = computeCommandEncoder.beginComputePass();
    computePassEncoder.setPipeline(forwardPipeline);
    computePassEncoder.setBindGroup(0, this.forwardBindGroup0);
    for (let li = 1; li < this.layersNum - 1; li++) {
      computePassEncoder.setBindGroup(1, this.bindGroup1s[li]);
      computePassEncoder.dispatchWorkgroups(Math.ceil(this.layerDatas[li].neuronsNum / 64), 1, 1);
    }
    {
      // 出力層
      const li = this.layersNum - 1;
      const layerData = this.layerDatas[li];
      computePassEncoder.setBindGroup(1, this.bindGroup1s[li]);
      computePassEncoder.dispatchWorkgroups(Math.ceil(this.layerDatas[li].neuronsNum / 64), 1, 1); // 一回汎用シェーダーを実行
      if (layerData.activationFunction === ActivationFunction.Softmax) { // softmax専用の処理
        computePassEncoder.setPipeline(forwardSoftmaxSumPipeline);
        computePassEncoder.setBindGroup(2, this.forwardSoftmaxBindGroup);
        computePassEncoder.dispatchWorkgroups(1, 1, 1);
        computePassEncoder.setPipeline(forwardSoftmaxDivPipeline);
        computePassEncoder.dispatchWorkgroups(Math.ceil(layerData.neuronsNum / 64), 1, 1);
      }
    }
    computePassEncoder.end();
    device.queue.submit([computeCommandEncoder.finish()]);
  }

  public runAndLearning(input: GPUAllowSharedBufferSource, target: GPUAllowSharedBufferSource): void {
    device.queue.writeBuffer(this.activatedValueBuffer, 0, input);
    device.queue.writeBuffer(this.targetBuffer, 0, target);
    const computeCommandEncoder = device.createCommandEncoder();
    const computePassEncoder = computeCommandEncoder.beginComputePass();
    computePassEncoder.setPipeline(forwardPipeline);
    computePassEncoder.setBindGroup(0, this.forwardBindGroup0);
    for (let li = 1; li < this.layersNum - 1; li++) {
      computePassEncoder.setBindGroup(1, this.bindGroup1s[li]);
      computePassEncoder.dispatchWorkgroups(Math.ceil(this.layerDatas[li].neuronsNum / 64), 1, 1);
    }
    {
      // 出力層
      const li = this.layersNum - 1;
      const layerData = this.layerDatas[li];
      computePassEncoder.setBindGroup(1, this.bindGroup1s[li]);
      computePassEncoder.dispatchWorkgroups(Math.ceil(this.layerDatas[li].neuronsNum / 64), 1, 1); // 一回汎用シェーダーを実行
      if (layerData.activationFunction === ActivationFunction.Softmax) { // softmax専用の処理
        computePassEncoder.setPipeline(forwardSoftmaxSumPipeline);
        computePassEncoder.setBindGroup(2, this.forwardSoftmaxBindGroup);
        computePassEncoder.dispatchWorkgroups(1, 1, 1);
        computePassEncoder.setPipeline(forwardSoftmaxDivPipeline);
        computePassEncoder.dispatchWorkgroups(Math.ceil(layerData.neuronsNum / 64), 1, 1);
      }
    }

    // 逆伝播
    computePassEncoder.setBindGroup(0, this.backwardBindGroup0);
    computePassEncoder.setBindGroup(2, this.backwardBindGroup2);
    computePassEncoder.setPipeline(backwardOutputPipeline);
    {
      // 出力層
      const li = this.layersNum - 1;
      computePassEncoder.setBindGroup(1, this.bindGroup1s[li]);
      computePassEncoder.dispatchWorkgroups(Math.ceil(this.layerDatas[li].neuronsNum / 64), 1, 1);
    }
    computePassEncoder.setPipeline(backwardHiddenPipeline);
    for (let li = this.layersNum - 2; li >= 1; li--) {
      computePassEncoder.setBindGroup(1, this.bindGroup1s[li]);
      computePassEncoder.dispatchWorkgroups(Math.ceil(this.layerDatas[li].neuronsNum / 64), 1, 1);
    }
    computePassEncoder.end();
    computeCommandEncoder.copyBufferToBuffer(this.newWeightBuffer, 0, this.weightBuffer, 0, this.newWeightBuffer.size);
    device.queue.submit([computeCommandEncoder.finish()]);
  }

  public runForLearning(input: GPUAllowSharedBufferSource): void {
    device.queue.writeBuffer(this.activatedValueBuffer, 0, input);
    const computeCommandEncoder = device.createCommandEncoder();
    const computePassEncoder = computeCommandEncoder.beginComputePass();
    computePassEncoder.setPipeline(forwardPipeline);
    computePassEncoder.setBindGroup(0, this.forwardBindGroup0);
    for (let li = 1; li < this.layersNum - 1; li++) {
      computePassEncoder.setBindGroup(1, this.bindGroup1s[li]);
      computePassEncoder.dispatchWorkgroups(Math.ceil(this.layerDatas[li].neuronsNum / 64), 1, 1);
    }
    {
      // 出力層
      const li = this.layersNum - 1;
      const layerData = this.layerDatas[li];
      computePassEncoder.setBindGroup(1, this.bindGroup1s[li]);
      computePassEncoder.dispatchWorkgroups(Math.ceil(this.layerDatas[li].neuronsNum / 64), 1, 1); // 一回汎用シェーダーを実行
      if (layerData.activationFunction === ActivationFunction.Softmax) { // softmax専用の処理
        computePassEncoder.setPipeline(forwardSoftmaxSumPipeline);
        computePassEncoder.setBindGroup(2, this.forwardSoftmaxBindGroup);
        computePassEncoder.dispatchWorkgroups(1, 1, 1);
        computePassEncoder.setPipeline(forwardSoftmaxDivPipeline);
        computePassEncoder.dispatchWorkgroups(Math.ceil(layerData.neuronsNum / 64), 1, 1);
      }
    }
    computePassEncoder.end();
    computeCommandEncoder.copyBufferToBuffer(this.newWeightBuffer, 0, this.weightBuffer, 0, this.newWeightBuffer.size);
    device.queue.submit([computeCommandEncoder.finish()]);
  }

  public learning(target: GPUAllowSharedBufferSource): void {
    device.queue.writeBuffer(this.targetBuffer, 0, target);
    const computeCommandEncoder = device.createCommandEncoder();
    const computePassEncoder = computeCommandEncoder.beginComputePass();
    // 逆伝播
    computePassEncoder.setBindGroup(0, this.backwardBindGroup0);
    computePassEncoder.setBindGroup(2, this.backwardBindGroup2);
    computePassEncoder.setPipeline(backwardOutputPipeline);
    {
      // 出力層
      const li = this.layersNum - 1;
      computePassEncoder.setBindGroup(1, this.bindGroup1s[li]);
      computePassEncoder.dispatchWorkgroups(Math.ceil(this.layerDatas[li].neuronsNum / 64), 1, 1);
    }
    computePassEncoder.setPipeline(backwardHiddenPipeline);
    for (let li = this.layersNum - 2; li >= 1; li--) {
      computePassEncoder.setBindGroup(1, this.bindGroup1s[li]);
      computePassEncoder.dispatchWorkgroups(Math.ceil(this.layerDatas[li].neuronsNum / 64), 1, 1);
    }
    computePassEncoder.end();
    computeCommandEncoder.copyBufferToBuffer(this.newWeightBuffer, 0, this.weightBuffer, 0, this.newWeightBuffer.size);
    device.queue.submit([computeCommandEncoder.finish()]);
  }
}