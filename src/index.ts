import {
  loadGraphModel,
  browser as tfBrowser,
  dispose,
  slice,
  sub,
  div,
  add,
  squeeze,
  concat,
  type Tensor3D,
  GraphModel,
} from "@tensorflow/tfjs";

function transformCoordinate(
  coord: number,
  scale: number,
  offset: number
): number {
  // Transform coordinates from model space back to original image space
  return coord * scale - offset;
}

function scalePrediction(
  prediction: {
    box: [number, number, number, number];
    score: number;
    keypoints: [number, number, number][];
  },
  transformParams: {
    scale: number;
    xOffset: number;
    yOffset: number;
  }
) {
  const { scale, xOffset, yOffset } = transformParams;

  // Transform bounding box coordinates
  const [modelX1, modelY1, modelX2, modelY2] = prediction.box;
  const scaledBox: [number, number, number, number] = [
    transformCoordinate(modelX1, scale, xOffset),
    transformCoordinate(modelY1, scale, yOffset),
    transformCoordinate(modelX2, scale, xOffset),
    transformCoordinate(modelY2, scale, yOffset),
  ];

  // Transform keypoint coordinates
  const scaledKeypoints: [number, number, number][] = prediction.keypoints.map(
    ([x, y, confidence]) => [
      transformCoordinate(x, scale, xOffset),
      transformCoordinate(y, scale, yOffset),
      confidence, // Confidence stays the same
    ]
  );

  return {
    box: scaledBox,
    score: prediction.score,
    keypoints: scaledKeypoints,
  };
}

function renderPrediction(props: {
  canvas: HTMLCanvasElement;
  score: number;
  box: [number, number, number, number];
  keypoints: [number, number, number][];
  source: HTMLImageElement;
  width: number;
  height: number;
}) {
  const threshold = 0.5;
  const { canvas, box, keypoints, source, score, width, height } = props;

  if (score < threshold) {
    return;
  }

  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(source, 0, 0, width, height);

  // Draw bounding box (coordinates are already transformed)
  const [x1, y1, x2, y2] = box;
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  // Draw keypoints (coordinates are already transformed)
  for (const [x, y, confidence] of keypoints) {
    if (confidence > threshold) {
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fillStyle = "red";
      ctx.fill();
    }
  }
}

async function getBestPrediction(predictions: Tensor3D): Promise<{
  box: [number, number, number, number];
  score: number;
  keypoints: [number, number, number][];
}> {
  // Reshape predictions from [1, 56, 8400] to [1, 8400, 56] for easier processing
  // Each of the 8400 predictions now has 56 values: [x, y, w, h, conf, kpt1_x, kpt1_y, kpt1_c, ...]
  const reshapedPredictions = predictions.transpose([0, 2, 1]);

  // Extract bounding box components (center format: x, y, width, height)
  const centerX = slice(reshapedPredictions, [0, 0, 0], [-1, -1, 1]);
  const centerY = slice(reshapedPredictions, [0, 0, 1], [-1, -1, 1]);
  const width = slice(reshapedPredictions, [0, 0, 2], [-1, -1, 1]);
  const height = slice(reshapedPredictions, [0, 0, 3], [-1, -1, 1]);

  // Convert from center format to corner format (x1, y1, x2, y2)
  const halfWidth = div(width, 2);
  const halfHeight = div(height, 2);
  const x1 = sub(centerX, halfWidth);
  const y1 = sub(centerY, halfHeight);
  const x2 = add(centerX, halfWidth);
  const y2 = add(centerY, halfHeight);

  // Extract confidence scores and keypoints
  const confidenceScores = slice(reshapedPredictions, [0, 0, 4], [-1, -1, 1]);
  const allKeypoints = slice(reshapedPredictions, [0, 0, 5], [-1, -1, -1]); // All remaining 51 values (17 keypts × 3)

  // Find the prediction with highest confidence
  const scoresArray = await confidenceScores.data();
  const bestPredictionIndex = scoresArray.indexOf(Math.max(...scoresArray));
  const bestConfidence = scoresArray[bestPredictionIndex];

  // Extract the best bounding box [x1, y1, x2, y2]
  const bestBoundingBox = squeeze(
    concat(
      [
        slice(x1, [0, bestPredictionIndex, 0], [1, 1, 1]),
        slice(y1, [0, bestPredictionIndex, 0], [1, 1, 1]),
        slice(x2, [0, bestPredictionIndex, 0], [1, 1, 1]),
        slice(y2, [0, bestPredictionIndex, 0], [1, 1, 1]),
      ],
      2
    )
  );

  // Extract the best keypoints (51 values: 17 keypoints × 3 values each)
  const bestKeypointsTensor = squeeze(
    slice(allKeypoints, [0, bestPredictionIndex, 0], [1, 1, -1])
  );

  // Convert keypoints tensor to array and group into [x, y, confidence] triplets
  const keypointsData = [...(await bestKeypointsTensor.data())];
  const formattedKeypoints: [number, number, number][] = [];

  for (let i = 0; i < keypointsData.length; i += 3) {
    const x = keypointsData[i];
    const y = keypointsData[i + 1];
    const confidence = keypointsData[i + 2];
    formattedKeypoints.push([x, y, confidence]);
  }
  const boxData = [...(await bestBoundingBox.data())] as [
    number,
    number,
    number,
    number
  ];

  // Clean up tensors to prevent memory leaks
  dispose([
    reshapedPredictions,
    centerX,
    centerY,
    width,
    height,
    halfWidth,
    halfHeight,
    x1,
    y1,
    x2,
    y2,
    confidenceScores,
    allKeypoints,
    bestBoundingBox,
    bestKeypointsTensor,
  ]);

  return {
    box: boxData,
    score: bestConfidence,
    keypoints: formattedKeypoints,
  };
}

/**
 * Apply letterboxing to maintain aspect ratio when resizing for model input
 * Letterboxing adds padding (black bars) to make the image square before resizing
 * This prevents distortion that would occur with direct resizing
 */
function processImageWithLetterboxing(
  image: HTMLImageElement,
  model: GraphModel
) {
  // Get model's expected input dimensions
  const modelInputShape = model.inputs[0].shape!;
  const [modelHeight, modelWidth] = modelInputShape.slice(1, 3);

  // Convert image to tensor and normalize to [0, 1]
  const originalImageTensor = tfBrowser.fromPixels(image).toFloat().div(255);

  // Step 1: Calculate letterboxing parameters
  // Find the larger dimension to determine the target square size
  const originalWidth = image.width;
  const originalHeight = image.height;
  const targetSquareSize = Math.max(originalWidth, originalHeight);

  // Calculate how much padding is needed on each axis
  const totalWidthPadding = targetSquareSize - originalWidth;
  const totalHeightPadding = targetSquareSize - originalHeight;

  // Distribute padding equally on both sides (center the image)
  const leftPadding = Math.floor(totalWidthPadding / 2);
  const rightPadding = totalWidthPadding - leftPadding;
  const topPadding = Math.floor(totalHeightPadding / 2);
  const bottomPadding = totalHeightPadding - topPadding;

  console.log(`Original: ${originalWidth}x${originalHeight}`);
  // Original: 250x405
  console.log(`Target square: ${targetSquareSize}x${targetSquareSize}`);
  // Target square: 405x405
  console.log(
    `Padding: top=${topPadding}, bottom=${bottomPadding}, left=${leftPadding}, right=${rightPadding}`
  );
  // Padding: top=0, bottom=0, left=77, right=78

  // Step 2: Apply letterboxing by adding padding (creates black bars)
  const letterboxedImage = originalImageTensor.pad<Tensor3D>([
    [topPadding, bottomPadding], // Height padding
    [leftPadding, rightPadding], // Width padding
    [0, 0], // No channel padding
  ]);

  // Step 3: Resize the square image to model input size
  const resizedImage = letterboxedImage.resizeBilinear<Tensor3D>([
    modelHeight,
    modelWidth,
  ]);

  // Step 4: Add batch dimension for model input [1, height, width, channels]
  const batchedImage = resizedImage.expandDims(0);

  // Calculate transformation parameters for mapping predictions back to original coordinates
  const scale = targetSquareSize / modelWidth; // Scale factor to map from model space back to letterboxed space
  const xOffset = leftPadding; // X displacement caused by letterboxing
  const yOffset = topPadding; // Y displacement caused by letterboxing

  console.log(
    `Transformation - scale: ${scale}, xOffset: ${xOffset}, yOffset: ${yOffset}`
  );
  // Transformation - scale: 0.6328125, xOffset: 77, yOffset: 0

  // Clean up intermediate tensors
  dispose([originalImageTensor, letterboxedImage, resizedImage]);

  return {
    processedImage: batchedImage,
    transformParams: {
      scale,
      xOffset,
      yOffset,
      originalWidth,
      originalHeight,
    },
  };
}

async function main() {
  const image = document.querySelector("img")!;
  const canvas = document.querySelector("canvas")!;
  const modelURL =
    import.meta.env.BASE_URL + "models/yolov8n-pose_web_model/model.json";
  const model = await loadGraphModel(modelURL);
  const { processedImage, transformParams } = processImageWithLetterboxing(
    image,
    model
  );
  const predictions = model.predict(processedImage) as Tensor3D;
  const bestPrediction = await getBestPrediction(predictions);
  const scaledPrediction = scalePrediction(bestPrediction, transformParams);
  renderPrediction({
    canvas,
    source: image,
    ...scaledPrediction,
    width: transformParams.originalWidth,
    height: transformParams.originalHeight,
  });
  dispose([processedImage, predictions]);
}
main();
