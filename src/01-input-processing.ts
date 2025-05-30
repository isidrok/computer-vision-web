import {
  loadGraphModel,
  browser as tfBrowser,
  type Tensor3D,
  GraphModel,
  dispose,
} from "@tensorflow/tfjs";

async function main() {
  const model = await loadGraphModel(
    "/models/yolov8n-pose_web_model/model.json"
  );
  console.log("Model input shape:", model.inputs[0].shape);
  const images = document.querySelectorAll("img");
  images.forEach((image) => {
    console.log("Processing image", image.getAttribute("src"));
    processImage(
      model,
      image as HTMLImageElement,
      image.nextElementSibling as HTMLCanvasElement
    );
  });
}

function processImage(
  model: GraphModel,
  image: HTMLImageElement,
  canvas: HTMLCanvasElement
) {
  const [inputHeight, inputWidth] = model.inputs[0].shape!.slice(1, 3);
  canvas.width = inputWidth;
  canvas.height = inputHeight;
  const imageTensor = tfBrowser.fromPixels(image);
  console.log("Original image shape:", imageTensor.shape);
  const resizedImageTensor = imageTensor
    .toFloat()
    .div(255)
    .resizeBilinear<Tensor3D>([inputHeight, inputWidth]);
  console.log("Resized image shape:", resizedImageTensor.shape);
  tfBrowser.toPixels(resizedImageTensor, canvas);
  const finalImageTensor = resizedImageTensor.expandDims(0);
  console.log("Final image shape:", finalImageTensor.shape);
  dispose([imageTensor, resizedImageTensor, finalImageTensor]);
}

main();
