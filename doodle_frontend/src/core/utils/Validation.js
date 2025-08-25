export class Validation {
  static hasDrawing(normalizedImageArray, threshold = 0.1) {
    return Array.isArray(normalizedImageArray) && normalizedImageArray.some(p => p > threshold);
  }
}

export default Validation;
