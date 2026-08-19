# NeuralNetworkGPU
学習用の src/mnist.json は別でダウンロードするか以下のpythonを実行して作ってください
```
from tensorflow.keras.datasets import mnist
import json

(x_train, y_train), _ = mnist.load_data()

images = []
labels = []

for img, label in zip(x_train, y_train):
  pixels = img.flatten()

  normalized = [float(p) / 255 for p in pixels]

  images.append(normalized)
  labels.append(int(label))

data = {
  "width": 28,
  "height": 28,
  "valueRangeMin": 0,
  "valueRangeMax": 1,
  "labels": labels,
  "images": images,
}

with open("mnist.json", "w", encoding="utf-8") as f:
  json.dump(data, f, separators=(",", ":"))

print("saved mnist.json")
```
必要なパッケージ
```
pip install tensorflow
pip3 install tensorflow
```