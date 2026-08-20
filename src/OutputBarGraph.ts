interface OutputBarGraphOptions {
  padding?: number;
  barColor?: string;
  highlightColor?: string; // 最大値のバーを強調する色
  axisColor?: string;
  textColor?: string;
  backgroundColor?: string;
  barGap?: number; // バー同士の隙間(px)
  labels?: string[]; // 各出力のラベル(例: ['0','1',...,'9'])
}

export class OutputBarGraph {
  private ctx: CanvasRenderingContext2D;
  private options: Required<OutputBarGraphOptions>;

  constructor(
    private canvas: HTMLCanvasElement,
    options: OutputBarGraphOptions = {}
  ) {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D context not available');
    this.ctx = ctx;

    this.options = {
      padding: options.padding ?? 24,
      barColor: options.barColor ?? '#4f9dff',
      highlightColor: options.highlightColor ?? '#ff6b4f',
      axisColor: options.axisColor ?? '#888888',
      textColor: options.textColor ?? '#dddddd',
      backgroundColor: options.backgroundColor ?? '#111111',
      barGap: options.barGap ?? 6,
      labels: options.labels ?? [],
    };
  }

  /**
   * @param values 各クラスの出力値(softmax後の確率、0〜1想定。それ以外でも自動スケールされる)
   */
  draw(values: number[]): void {
    if (values.length === 0) return;

    const { ctx, canvas, options } = this;
    const { padding, barColor, highlightColor, axisColor, textColor, backgroundColor, barGap, labels } = options;

    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = backgroundColor;
    ctx.fillRect(0, 0, width, height);

    // ラベル表示分の左余白を確保
    const labelAreaWidth = labels.length > 0 ? 30 : 0;
    const graphX = padding + labelAreaWidth;
    const graphW = width - graphX - padding;
    const graphH = height - padding * 2;

    const n = values.length;
    const barHeight = (graphH - barGap * (n - 1)) / n;

    // スケール: 0〜1想定だが、念のため最大値でクランプ
    const maxVal = Math.max(...values, 1e-8);
    const scaleMax = Math.max(1.0, maxVal); // softmax出力なら基本1で固定表示、稀に超える場合だけ広げる

    const maxIndex = values.indexOf(Math.max(...values));

    ctx.textBaseline = 'middle';
    ctx.font = '12px sans-serif';

    for (let i = 0; i < n; i++) {
      const y = padding + i * (barHeight + barGap);
      const barW = Math.max(0, (values[i] / scaleMax) * graphW);

      // バー本体
      ctx.fillStyle = i === maxIndex ? highlightColor : barColor;
      ctx.fillRect(graphX, y, barW, barHeight);

      // ラベル(左側)
      if (labels.length > 0) {
        ctx.fillStyle = textColor;
        ctx.textAlign = 'right';
        ctx.fillText(labels[i] ?? String(i), graphX - 6, y + barHeight / 2);
      }

      // 数値(バーの右端 or バー内)
      ctx.fillStyle = textColor;
      ctx.textAlign = 'left';
      const valueText = values[i].toFixed(3);
      const textX = barW + 4 < graphW - 40 ? graphX + barW + 4 : graphX + barW - 40;
      ctx.fillText(valueText, textX, y + barHeight / 2);
    }

    // 軸線
    ctx.strokeStyle = axisColor;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(graphX, padding);
    ctx.lineTo(graphX, height - padding);
    ctx.stroke();
  }
}