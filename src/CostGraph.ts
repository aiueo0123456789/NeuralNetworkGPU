interface CostGraphOptions {
  padding?: number;
  lineColor?: string;
  axisColor?: string;
  backgroundColor?: string;
  maxPoints?: number;
}

export class CostGraph {
  private ctx: CanvasRenderingContext2D;
  private options: Required<CostGraphOptions>;

  constructor(
    private canvas: HTMLCanvasElement,
    options: CostGraphOptions = {}
  ) {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D context not available');
    this.ctx = ctx;

    this.options = {
      padding: options.padding ?? 24,
      lineColor: options.lineColor ?? '#4f9dff',
      axisColor: options.axisColor ?? '#888888',
      backgroundColor: options.backgroundColor ?? '#111111',
      maxPoints: options.maxPoints ?? 500,
    };
  }

  draw(costs: number[]): void {
    if (costs.length === 0) return;

    const { ctx, canvas, options } = this;
    const { padding, lineColor, axisColor, backgroundColor, maxPoints } = options;

    // 点が多すぎる場合は間引いて表示(直近の傾向が見やすいように等間隔サンプリング)
    const data = this.downsample(costs, maxPoints);

    const width = canvas.width;
    const height = canvas.height;

    // 背景クリア
    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    const graphW = width - padding * 2;
    const graphH = height - padding * 2;

    const minCost = Math.min(...data);
    const maxCost = Math.max(...data);
    // 全部同じ値の場合など、範囲が0にならないようにする
    const range = maxCost - minCost || 1;

    // 軸
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(padding, padding);
    ctx.lineTo(padding, height - padding);
    ctx.lineTo(width - padding, height - padding);
    ctx.stroke();

    // Y軸ラベル(min / max)
    ctx.fillStyle = axisColor;
    ctx.font = '10px sans-serif';
    ctx.textBaseline = 'middle';
    ctx.fillText(maxCost.toFixed(4), 2, padding);
    ctx.fillText(minCost.toFixed(4), 2, height - padding);

    // 誤差推移の折れ線
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.5;
    ctx.beginPath();

    data.forEach((cost, i) => {
      const x = padding + (i / (data.length - 1 || 1)) * graphW;
      const normalized = (cost - minCost) / range; // 0(min)〜1(max)
      const y = height - padding - normalized * graphH;

      if (i === 0) {
        ctx.moveTo(x, y);
      } else {
        ctx.lineTo(x, y);
      }
    });

    ctx.stroke();

    // 最新値をテキストで表示
    const latest = costs[costs.length - 1];
    ctx.fillStyle = lineColor;
    ctx.fillText(`cost: ${latest.toFixed(6)}`, width - padding - 100, padding - 10);
  }

  private downsample(costs: number[], maxPoints: number): number[] {
    if (costs.length <= maxPoints) return costs;

    const result: number[] = [];
    const step = costs.length / maxPoints;
    for (let i = 0; i < maxPoints; i++) {
      result.push(costs[Math.floor(i * step)]);
    }
    return result;
  }
}