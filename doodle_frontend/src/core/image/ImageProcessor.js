// ImageProcessor: converts canvas content to ML-friendly image arrays
export class ImageProcessor {
  /**
   * Get 28x28 grayscale normalized image data from a source canvas
   * Returns { image: number[], width: 28, height: 28 }
   */
  static to28x28Grayscale(canvas) {
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 28;
    tempCanvas.height = 28;

    // White background
    tempCtx.fillStyle = '#FFFFFF';
    tempCtx.fillRect(0, 0, 28, 28);

    // Scale original
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);

    // Extract and normalize
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
    const normalizedData = [];
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      const gray = (r + g + b) / 3; // 0..255
      const normalized = (255 - gray) / 255; // invert + normalize
      normalizedData.push(normalized);
    }

    return {
      image: normalizedData,
      width: 28,
      height: 28,
    };
  }
}

export default ImageProcessor;
