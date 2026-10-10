import { CostGraph } from "./CostGraph";
import { DrawUI } from "./DrawUI";
import { ActivationFunction, NeuralNetworkGPU } from "./neuralNetworkGPU";
import { OutputBarGraph } from "./OutputBarGraph";

const costCanvas = document.getElementById('costCanvas') as HTMLCanvasElement;
const outputBarCanvas = document.getElementById('outputBarCanvas') as HTMLCanvasElement;
const drawCanvas = document.getElementById('drawCanvas') as HTMLCanvasElement;
const costGraph = new CostGraph(costCanvas, { maxPoints: 5000 });
const outputBarGraph = new OutputBarGraph(outputBarCanvas, {
  labels: ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
});
const drawUI = new DrawUI(drawCanvas);

const neuralNetworkGPU = new NeuralNetworkGPU({
  inputLayer: {
    neuronsNum: 28 * 28,
    activationFunction: ActivationFunction.None
  },
  hiddenLayers: [
    {
      neuronsNum: 256,
      activationFunction: ActivationFunction.LeakyReLU
    },
    {
      neuronsNum: 128,
      activationFunction: ActivationFunction.LeakyReLU
    },
    {
      neuronsNum: 64,
      activationFunction: ActivationFunction.LeakyReLU
    },
    {
      neuronsNum: 32,
      activationFunction: ActivationFunction.LeakyReLU
    }
  ],
  outputLayer: {
    neuronsNum: 10,
    // activationFunction: ActivationFunction.Sigmoid
    activationFunction: ActivationFunction.Softmax
  }
});

neuralNetworkGPU.setRandom(1);
neuralNetworkGPU.setLearningRateBuffer(0.0001);

const testData: {labels: any[], images: any[] } = await fetch(`src/data.json`).then((x) => x.json());
testData.images = testData.images.map(image => new Float32Array(image));
const testDatasNum = testData.labels.length;
console.log(`${testDatasNum} のデータを学習します`)

const targetDatas: Float32Array[] = [];
for (let i = 0; i < 10; i ++) {
  const array = new Float32Array(10);
  array[i] = 1;
  targetDatas.push(array);
}

const costs: number[] = [];

const maxLearningCount = 500000;
let learningCount = 0;
let loopNum = 0;
async function updateCosts(): Promise<void> {
  let sumCost = 0;
  for (let i = 0; i < 10; i ++) {
    let testDataIndex = Math.floor(Math.random() * testDatasNum);
    const label = testData.labels[testDataIndex];
    const image = testData.images[testDataIndex];
    drawUI.setPixels(image);
    sumCost += await neuralNetworkGPU.getCost(image, targetDatas[label]);
  }
  costs.push(sumCost / 10);
  costGraph.draw(costs);
}

function update(): void {
  for (let i = 0; i < 100; i ++) {
    if (maxLearningCount <= learningCount) break ;
    let testDataIndex = Math.floor(Math.random() * testDatasNum);
    const label = testData.labels[testDataIndex];
    const image = testData.images[testDataIndex];
    neuralNetworkGPU.learning(image, targetDatas[label]);
    learningCount++;
  }
  if (loopNum % 10 == 0) updateCosts();
  loopNum ++;
  if (learningCount < maxLearningCount) {
    requestAnimationFrame(update);
  } else {
    console.log("学習終了")
  }
}

update();

function update2(): void {
  drawUI.draw();
  requestAnimationFrame(update2);
}
update2();

async function pixelsUpdate(): Promise<void> {
  neuralNetworkGPU.run(drawUI.pixels);
  outputBarGraph.draw(Array.from(await neuralNetworkGPU.getAnswer()));
}

let isMousedown = false;
drawCanvas.addEventListener("mousedown", (e) => {
  isMousedown = true;
})
drawCanvas.addEventListener("mouseup", (e) => {
  isMousedown = false;
})
drawCanvas.addEventListener("mousemove", (e) => {
  if (!isMousedown) return ;
  const rect = drawCanvas.getBoundingClientRect();
  const cvsK = drawCanvas.height / rect.height;
  const mouseX = (e.clientX - rect.left) * cvsK;
  const mouseY = (e.clientY - rect.top) * cvsK;
  if (
    mouseX > 0 &&
    mouseX < drawCanvas.width &&
    mouseY > 0 &&
    mouseY < drawCanvas.height
  ) {
    drawUI.setPixel(mouseX, mouseY, 1);
    pixelsUpdate();
  }
})
document.getElementById("reset")?.addEventListener("click", () => {
  drawUI.clear();
})
document.getElementById("set")?.addEventListener("click", () => {
  let testDataIndex = Math.floor(Math.random() * testDatasNum);
  const image = testData.images[testDataIndex];
  drawUI.setPixels(image);
  pixelsUpdate();
})