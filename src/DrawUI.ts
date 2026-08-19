export class DrawUI {
  private ctx: CanvasRenderingContext2D;
  private width: number;
  private height: number;
  public pixels: Float32Array;

  constructor(
    private canvas: HTMLCanvasElement,
  ) {
    this.width = 28;
    this.height = 28;
    canvas.width = 28;
    canvas.height = 28;
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('2D context not available');
    this.ctx = ctx;

    this.pixels = new Float32Array(this.width * this.height);
  }

  setPixels(pixels: any) {
    this.pixels.set(pixels);
  }

  clear() {
    this.pixels.fill(0);
  }

  setPixel(x: number,y: number,value: number) {
    this.pixels[Math.floor(x) + Math.floor(y) * this.width] = value;
  }

  draw(): void {
    const { ctx, canvas } = this;

    // 背景クリア
    ctx.fillStyle = "rgb(0,0,0)";
    ctx.fillRect(0, 0, this.width, this.height);

    for (let y = 0; y < this.height; y ++) {
      for (let x = 0; x < this.width; x ++) {
        const pixelColor = this.pixels[x + y * this.width] * 255;
        if (pixelColor > 0) {
          ctx.fillStyle = `rgb(${pixelColor},${pixelColor},${pixelColor})`;
          ctx.beginPath();
          ctx.arc(x, y, 1, 0, Math.PI * 2);
          ctx.fill();
        }
      }
    }
  }
}