import {
  loadGraphModel,
  browser as tfBrowser,
  tidy,
  slice,
  sub,
  div,
  add,
  squeeze,
  concat,
  type Tensor3D,
  GraphModel,
} from "@tensorflow/tfjs";

/** Bounding box in corner format: [x1, y1, x2, y2] */
type BoundingBox = [number, number, number, number];

/** Keypoint with coordinates and confidence: [x, y, confidence] */
type Keypoint = [number, number, number];

/** Model prediction containing bounding box, confidence score, and keypoints */
interface Prediction {
  box: BoundingBox;
  score: number;
  keypoints: Keypoint[];
}

/** Parameters for transforming coordinates from model space to original image space */
interface TransformParams {
  scale: number;
  xOffset: number;
  yOffset: number;
  originalWidth: number;
  originalHeight: number;
}

/** Result of image processing with letterboxing */
interface ProcessedImageResult {
  processedImage: Tensor3D;
  transformParams: TransformParams;
}

/** Props for rendering predictions on canvas */
interface RenderPredictionProps {
  canvas: HTMLCanvasElement;
  score: number;
  box: BoundingBox;
  keypoints: Keypoint[];
  source: HTMLImageElement | HTMLVideoElement;
  width: number;
  height: number;
}

/**
 * Transform a single coordinate from model space back to original image space.
 *
 * @param coord - Coordinate value in model space
 * @param scale - Scale factor to convert from model space to letterboxed space
 * @param offset - Offset to subtract after scaling to account for letterboxing padding
 * @returns Transformed coordinate in original image space
 */
function transformCoordinate(
  coord: number,
  scale: number,
  offset: number
): number {
  return coord * scale - offset;
}

/**
 * Scale prediction coordinates from model space back to original image space.
 * This accounts for both the letterboxing padding and the resize operation.
 *
 * @param prediction - Raw prediction from the model with coordinates in model space
 * @param transformParams - Transformation parameters including scale and offsets
 * @returns Prediction with coordinates transformed to original image space
 */
function scalePrediction(
  prediction: Prediction,
  transformParams: TransformParams
): Prediction {
  const { scale, xOffset, yOffset } = transformParams;

  // Transform bounding box coordinates
  const [modelX1, modelY1, modelX2, modelY2] = prediction.box;
  const scaledBox: BoundingBox = [
    transformCoordinate(modelX1, scale, xOffset),
    transformCoordinate(modelY1, scale, yOffset),
    transformCoordinate(modelX2, scale, xOffset),
    transformCoordinate(modelY2, scale, yOffset),
  ];

  // Transform keypoint coordinates
  const scaledKeypoints: Keypoint[] = prediction.keypoints.map(
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

/**
 * Render pose estimation prediction on a canvas.
 * Draws the original image, bounding box, and keypoints with confidence above threshold.
 *
 * @param props - Rendering configuration including canvas, prediction data, and source image
 */
function renderPrediction(props: RenderPredictionProps): void {
  const CONFIDENCE_THRESHOLD = 0.5;
  const { canvas, box, keypoints, source, score, width, height } = props;

  // Skip rendering if prediction confidence is too low
  if (score < CONFIDENCE_THRESHOLD) {
    return;
  }

  // Set up canvas
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  ctx.clearRect(0, 0, width, height);
  ctx.drawImage(source, 0, 0, width, height);

  // Draw bounding box (coordinates are already transformed to original image space)
  const [x1, y1, x2, y2] = box;
  ctx.strokeStyle = "lime";
  ctx.lineWidth = 2;
  ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

  // Draw keypoints (coordinates are already transformed to original image space)
  ctx.fillStyle = "red";
  for (const [x, y, confidence] of keypoints) {
    if (confidence > CONFIDENCE_THRESHOLD) {
      ctx.beginPath();
      ctx.arc(x, y, 5, 0, 2 * Math.PI);
      ctx.fill();
    }
  }
}

/**
 * Extract the best prediction from YOLOv8 pose model output.
 * Processes the raw model output tensor to find the prediction with highest confidence,
 * converts bounding box from center format to corner format, and formats keypoints.
 *
 * @param predictions - Raw model output tensor with shape [1, 56, 8400]
 *                     56 channels: [x, y, w, h, conf, kpt1_x, kpt1_y, kpt1_c, ..., kpt17_x, kpt17_y, kpt17_c]
 *                     8400 predictions: different potential detections across the image
 * @returns Promise resolving to the best prediction with bounding box, score, and keypoints
 */
function getBestPrediction(predictions: Tensor3D): Prediction {
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
  const scoresArray = confidenceScores.dataSync();
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
  const keypointsData = [...bestKeypointsTensor.dataSync()];
  const formattedKeypoints: Keypoint[] = [];

  for (let i = 0; i < keypointsData.length; i += 3) {
    const x = keypointsData[i];
    const y = keypointsData[i + 1];
    const confidence = keypointsData[i + 2];
    formattedKeypoints.push([x, y, confidence]);
  }

  const boxData = [...bestBoundingBox.dataSync()] as BoundingBox;

  return {
    box: boxData,
    score: bestConfidence,
    keypoints: formattedKeypoints,
  };
}

/**
 * Apply letterboxing to maintain aspect ratio when resizing for model input.
 * Letterboxing adds padding (black bars) to make the image square before resizing.
 * This prevents distortion that would occur with direct resizing of non-square images.
 *
 * @param image - Source image element to process
 * @param model - TensorFlow.js model to get input shape requirements
 * @returns Object containing the processed image tensor and transformation parameters
 *          needed to map predictions back to original image coordinates
 */
function processImageWithLetterboxing(
  source: HTMLImageElement | HTMLVideoElement,
  model: GraphModel
): ProcessedImageResult {
  // Get model's expected input dimensions
  const modelInputShape = model.inputs[0].shape!;
  const [modelHeight, modelWidth] = modelInputShape.slice(1, 3);

  // Convert image to tensor and normalize to [0, 1]
  const originalImageTensor = tfBrowser.fromPixels(source).toFloat().div(255);

  // Step 1: Calculate letterboxing parameters
  // Find the larger dimension to determine the target square size
  const originalWidth = source.width;
  const originalHeight = source.height;
  const targetSquareSize = Math.max(originalWidth, originalHeight);

  // Calculate how much padding is needed on each axis
  const totalWidthPadding = targetSquareSize - originalWidth;
  const totalHeightPadding = targetSquareSize - originalHeight;

  // Distribute padding equally on both sides (center the image)
  const leftPadding = Math.floor(totalWidthPadding / 2);
  const rightPadding = totalWidthPadding - leftPadding;
  const topPadding = Math.floor(totalHeightPadding / 2);
  const bottomPadding = totalHeightPadding - topPadding;

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

  return {
    processedImage: batchedImage as Tensor3D,
    transformParams: {
      scale,
      xOffset,
      yOffset,
      originalWidth,
      originalHeight,
    },
  };
}

/**
 * Initialize and start video capture from the user's camera.
 * Automatically adjusts dimensions for portrait mode by swapping width and height.
 * 
 * @param params - Configuration object for video setup
 * @param params.video - HTML video element to receive the camera stream
 * @param params.width - Desired video width in pixels
 * @param params.height - Desired video height in pixels
 * @returns Promise that resolves when video metadata is loaded and ready for use
 */
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
    video.addEventListener("loadedmetadata", () => resolve(), { once: true });
  });
}

/**
 * Stop video capture and release camera resources.
 * Properly cleans up the media stream to free the camera for other applications.
 * 
 * @param video - HTML video element with active camera stream to stop
 */
function stopVideo(video: HTMLVideoElement) {
  const stream = video.srcObject as MediaStream;
  stream.getTracks().forEach((track) => track.stop());
  video.srcObject = null;
}

/**
 * Process a single frame for pose estimation and render results.
 * This function runs the complete inference pipeline:
 * 1. Applies letterboxing to maintain aspect ratio
 * 2. Runs model inference to detect poses
 * 3. Extracts and scales the best prediction
 * 4. Renders results on canvas
 * 5. Schedules the next frame processing
 * 
 * Uses TensorFlow.js tidy() to automatically clean up intermediate tensors
 * and prevent memory leaks during continuous processing.
 * 
 * @param canvas - HTML canvas element where results will be rendered
 * @param source - Image or video element to process for pose detection
 * @param model - Loaded YOLOv8 pose estimation model for inference
 */
function processImage(
  canvas: HTMLCanvasElement,
  source: HTMLImageElement | HTMLVideoElement,
  model: GraphModel
): void {
  tidy(() => {
    // Process image with letterboxing to maintain aspect ratio
    const { processedImage, transformParams } = processImageWithLetterboxing(
      source,
      model
    );

    // Run model inference
    const predictions = model.predict(processedImage) as Tensor3D;

    // Extract best prediction and transform coordinates back to original image space
    const bestPrediction = getBestPrediction(predictions);
    const scaledPrediction = scalePrediction(bestPrediction, transformParams);

    // Render results on canvas
    renderPrediction({
      canvas,
      source,
      ...scaledPrediction,
      width: transformParams.originalWidth,
      height: transformParams.originalHeight,
    });

    // Schedule next frame processing
    requestAnimationFrame(() => {
      processImage(canvas, source, model);
    });
  });
}

/**
 * Main application function that orchestrates the entire pose estimation pipeline.
 * 
 * This function sets up the complete real-time pose estimation application:
 * 1. Initializes DOM elements (button, video, canvas)
 * 2. Loads the YOLOv8 pose estimation model
 * 3. Sets up event handlers for start/stop functionality
 * 4. Manages video capture and processing lifecycle
 * 
 * The application uses a toggle button to start/stop the camera and pose detection.
 * When running, it continuously processes video frames and displays pose keypoints
 * and bounding boxes on an overlay canvas.
 * 
 * Pipeline overview:
 * 1. Load YOLOv8 pose estimation model from public/models directory
 * 2. Process input video frames with letterboxing to maintain aspect ratio
 * 3. Run model inference to get pose predictions
 * 4. Extract the best prediction and scale coordinates back to original image space
 * 5. Render the results on canvas with bounding box and keypoints
 * 
 */
async function main(): Promise<void> {
  // Get DOM elements
  const toggle = document.querySelector("button")!;
  const video = document.querySelector("video")!;
  const canvas = document.querySelector("canvas")!;
  // Load YOLOv8 pose estimation model
  const modelURL =
    import.meta.env.BASE_URL + "models/yolov8n-pose_web_model/model.json";
  const model = await loadGraphModel(modelURL);
  let running = false;
  toggle.addEventListener("click", async () => {
    if (running) {
      toggle.textContent = "Start";
      stopVideo(video);
    } else {
      toggle.textContent = "Stop";
      const { width, height } = video.getBoundingClientRect();
      // Since we are not specifying the width and height of the canvas and video
      // in the html, we need to manually override this attributes to configure
      // the actual size of the canvas and video buffers
      canvas.width = width;
      canvas.height = height;
      video.width = width;
      video.height = height;
      await startVideo({ video, width, height });
      requestAnimationFrame(() => {
        processImage(canvas, video, model);
      });
    }
    running = !running;
  });
}

// Start the application
main();
