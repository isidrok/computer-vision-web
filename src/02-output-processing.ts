import {
  loadGraphModel,
  browser as tfBrowser,
  dispose,
  type Tensor3D,
  GraphModel,
  slice,
  sub,
  div,
  add,
  concat,
  squeeze,
} from "@tensorflow/tfjs";

function renderPosePredictions(props: {
  ctx: CanvasRenderingContext2D;
  keypoints: [number, number, number][];
  width: number;
  height: number;
  source: HTMLImageElement;
}) {
  const { ctx, keypoints, width, height, source } = props;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(source, 0, 0, width, height);
  for (const [x, y, c] of keypoints) {
    if (c > 0.5) {
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    }
  }
}

async function getBestPrediction(predictions: Tensor3D): Promise<{
  box: number[];
  score: number;
  keypoints: [number, number, number][];
}> {
  const transpose = predictions.transpose([0, 2, 1]);
  const w = slice(transpose, [0, 0, 2], [-1, -1, 1]);
  const h = slice(transpose, [0, 0, 3], [-1, -1, 1]);
  const x1 = sub(slice(transpose, [0, 0, 0], [-1, -1, 1]), div(w, 2));
  const y1 = sub(slice(transpose, [0, 0, 1], [-1, -1, 1]), div(h, 2));
  const x2 = add(x1, w);
  const y2 = add(y1, h);
  const scores = slice(transpose, [0, 0, 4], [-1, -1, 1]);
  const keypoints = slice(transpose, [0, 0, 5], [-1, -1, -1]);
  const scoresData = await scores.data();
  const maxScoreIndex = scoresData.indexOf(Math.max(...scoresData));
  const bestBox = squeeze(
    concat(
      [
        slice(x1, [0, maxScoreIndex, 0], [1, 1, 1]),
        slice(y1, [0, maxScoreIndex, 0], [1, 1, 1]),
        slice(x2, [0, maxScoreIndex, 0], [1, 1, 1]),
        slice(y2, [0, maxScoreIndex, 0], [1, 1, 1]),
      ],
      2
    )
  );
  const bestScore = scoresData[maxScoreIndex];
  const bestKeypoints = squeeze(
    slice(keypoints, [0, maxScoreIndex, 0], [1, 1, -1])
  );

  const keypointsData = [...(await bestKeypoints.data())];
  const keypointsFormatted: [number, number, number][] = [];
  for (let i = 0; i < keypointsData.length; i += 3) {
    keypointsFormatted.push([
      keypointsData[i],
      keypointsData[i + 1],
      keypointsData[i + 2],
    ]);
  }
  dispose([transpose, w, h, x1, y1, x2, y2, scores, keypoints]);
  return {
    box: [...(await bestBox.data())],
    score: bestScore,
    keypoints: keypointsFormatted,
  };
}

async function processImage(
  model: GraphModel,
  image: HTMLImageElement,
  imageCanvas: HTMLCanvasElement,
  modelCanvas: HTMLCanvasElement
) {
  const [modelHeight, modelWidth] = model.inputs[0].shape!.slice(1, 3);
  imageCanvas.width = image.width;
  imageCanvas.height = image.height;
  modelCanvas.width = modelWidth;
  modelCanvas.height = modelHeight;
  const inputTensor = tfBrowser
    .fromPixels(image)
    .toFloat()
    .div(255)
    .resizeBilinear<Tensor3D>([modelHeight, modelWidth]);
  tfBrowser.toPixels(inputTensor, modelCanvas);
  const batchInputTensor = inputTensor.expandDims(0);
  const predictions = model.predict(batchInputTensor) as Tensor3D;
  const bestPrediction = await getBestPrediction(predictions);
  renderPosePredictions({
    ctx: imageCanvas.getContext("2d")!,
    keypoints: bestPrediction?.keypoints ?? [],
    width: image.width,
    height: image.height,
    source: image,
  });
  renderPosePredictions({
    ctx: modelCanvas.getContext("2d")!,
    keypoints: bestPrediction?.keypoints ?? [],
    width: modelWidth,
    height: modelHeight,
    source: image,
  });

  dispose([inputTensor, batchInputTensor, predictions]);
}

async function main() {
  const model = await loadGraphModel(
    import.meta.env.BASE_URL + "models/yolov8n-pose_web_model/model.json"
  );
  const elements = document.querySelectorAll("details");
  elements.forEach((el) => {
    const image = el.querySelector("img")!;
    const imageCanvas = el.querySelector("[data-image]")!;
    const modelCanvas = el.querySelector("[data-model]")!;
    processImage(
      model,
      image as HTMLImageElement,
      imageCanvas as HTMLCanvasElement,
      modelCanvas as HTMLCanvasElement
    );
  });
}

main();
