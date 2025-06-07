import {
  loadGraphModel,
  browser as tfBrowser,
  tidy,
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
  source?: HTMLImageElement;
}) {
  const { ctx, keypoints, width, height, source } = props;
  if (source) {
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(source, 0, 0, width, height);
  }
  for (const [x, y, c] of keypoints) {
    if (c > 0.5) {
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    }
  }
}
function getBestPrediction(predictions: Tensor3D): {
  box: number[];
  score: number;
  keypoints: [number, number, number][];
} {
  const transpose = predictions.transpose([0, 2, 1]);
  const w = slice(transpose, [0, 0, 2], [-1, -1, 1]);
  const h = slice(transpose, [0, 0, 3], [-1, -1, 1]);
  const x1 = sub(slice(transpose, [0, 0, 0], [-1, -1, 1]), div(w, 2));
  const y1 = sub(slice(transpose, [0, 0, 1], [-1, -1, 1]), div(h, 2));
  const x2 = add(x1, w);
  const y2 = add(y1, h);
  const scores = slice(transpose, [0, 0, 4], [-1, -1, 1]);
  const keypoints = slice(transpose, [0, 0, 5], [-1, -1, -1]);
  const scoresData = scores.dataSync();
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

  const keypointsData = [...bestKeypoints.dataSync()];
  const keypointsFormatted: [number, number, number][] = [];
  for (let i = 0; i < keypointsData.length; i += 3) {
    keypointsFormatted.push([
      keypointsData[i],
      keypointsData[i + 1],
      keypointsData[i + 2],
    ]);
  }
  return {
    box: [...bestBox.dataSync()],
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
  tidy(() => {
    const [modelHeight, modelWidth] = model.inputs[0].shape!.slice(1, 3);
    imageCanvas.width = image.width;
    imageCanvas.height = image.height;
    modelCanvas.width = modelWidth;
    modelCanvas.height = modelHeight;
    const input = tfBrowser.fromPixels(image).toFloat().div(255);
    const maxDim = Math.max(image.width, image.height);
    const padWidth = maxDim - image.width;
    const padHeight = maxDim - image.height;
    const dx = Math.floor(padWidth / 2);
    const dy = Math.floor(padHeight / 2);

    const padded = input.pad<Tensor3D>([
      [dy, padHeight - dy],
      [dx, padWidth - dx],
      [0, 0],
    ]);

    const resized = padded.resizeBilinear<Tensor3D>([640, 640]);
    tfBrowser.toPixels(resized, modelCanvas);
    const batched = resized.expandDims(0);
    const predictions = model.predict(batched) as Tensor3D;
    const bestPrediction = getBestPrediction(predictions);

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
    });
  });
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
    console.log("Processing image", image.getAttribute("src"));
    processImage(
      model,
      image as HTMLImageElement,
      imageCanvas as HTMLCanvasElement,
      modelCanvas as HTMLCanvasElement
    );
  });
}

main();
