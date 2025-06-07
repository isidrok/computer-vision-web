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
  source: HTMLVideoElement;
  scale: number;
  dx: number;
  dy: number;
}) {
  const { ctx, keypoints, width, height, source, scale, dx, dy } = props;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(source, 0, 0, width, height);
  for (let [x, y, c] of keypoints) {
    if (c > 0.5) {
      x = x * scale;
      y = y * scale;
      x = x - dx;
      y = y - dy;
      x = Math.max(0, Math.min(x, width));
      y = Math.max(0, Math.min(y, height));
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

async function processFrame(
  model: GraphModel,
  video: HTMLVideoElement,
  canvas: HTMLCanvasElement
) {
  tidy(() => {
    const input = tfBrowser.fromPixels(video).toFloat().div(255);
    const [modelHeight, modelWidth] = model.inputs[0].shape!.slice(1, 3);
    const [inputHeight, inputWidth] = input.shape.slice(0, 2);
    const maxDim = Math.max(inputWidth, inputHeight);
    const padWidth = maxDim - inputWidth;
    const padHeight = maxDim - inputHeight;
    const dx = Math.floor(padWidth / 2);
    const dy = Math.floor(padHeight / 2);
    const scale = maxDim / modelWidth;
    const padded = input.pad<Tensor3D>([
      [dy, padHeight - dy],
      [dx, padWidth - dx],
      [0, 0],
    ]);
    const resized = padded.resizeBilinear<Tensor3D>([modelWidth, modelHeight]);
    const batched = resized.expandDims(0);
    const predictions = model.predict(batched) as Tensor3D;
    const bestPrediction = getBestPrediction(predictions);
    renderPosePredictions({
      ctx: canvas.getContext("2d")!,
      keypoints: bestPrediction?.keypoints ?? [],
      width: inputWidth,
      height: inputHeight,
      source: video,
      scale,
      dx,
      dy,
    });
  });
  return requestAnimationFrame(() => processFrame(model, video, canvas));
}

async function startVideo({
  video,
  width,
  height,
}: {
  video: HTMLVideoElement;
  width: number;
  height: number;
}) {
  if (window.innerHeight > window.innerWidth) {
    // Portrait mode: swap width and height
    [width, height] = [height, width];
  }
  const stream = await navigator.mediaDevices.getUserMedia({
    audio: false,
    video: {
      width,
      height,
      facingMode: "user",
    },
  });
  video.srcObject = stream;

  return new Promise<void>((resolve) => {
    video.addEventListener("loadeddata", () => resolve(), { once: true });
  });
}

function stopVideo(video: HTMLVideoElement) {
  const stream = video.srcObject as MediaStream;
  stream.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
}

async function main() {
  const model = await loadGraphModel(
    import.meta.env.BASE_URL + "models/yolov8n-pose_web_model/model.json"
  );
  const video = document.querySelector("video")!;
  const toggle = document.querySelector("button")!;
  const canvas = document.querySelector("canvas")!;
  let requestAnimationFrameId: number;
  toggle.addEventListener("click", async () => {
    if (video.srcObject) {
      toggle.textContent = "Start";
      cancelAnimationFrame(requestAnimationFrameId);
      stopVideo(video);
    } else {
      toggle.textContent = "Stop";
      const { width, height } = video.getBoundingClientRect();
      canvas.width = width;
      canvas.height = height;
      await startVideo({ video, width, height });
      requestAnimationFrameId = await processFrame(model, video, canvas);
    }
  });
}

main();
