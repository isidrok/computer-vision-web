import { defineConfig } from "vite";

export default defineConfig({
  base: "/computer-vision-web/",
  build: {
    rollupOptions: {
      input: {
        index: "index.html",
        inputProcessing: "01-input-processing.html",
        outputProcessing: "02-output-processing.html",
        letterboxInputProcessing: "03-letterbox-input-processing.html",
        letterboxOutputProcessing: "04-letterbox-output-processing.html",
        liveVideo: "05-live-video.html",
      },
    },
  },
});
