import Config from '../config/Config';

export class PredictionClient {
  static async predict(imageData) {
    const resp = await fetch(`${Config.BACKEND_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(imageData),
    });
    if (!resp.ok) {
      let errMsg = 'Prediction failed';
      try {
        const err = await resp.json().catch(() => ({}));
        if (err && (err.detail || err.message)) errMsg = err.detail || err.message;
      } catch {}
      throw new Error(errMsg);
    }
    return resp.json();
  }

  static async downloadProcessed(imageData) {
    const resp = await fetch(`${Config.BACKEND_URL}/download_processed`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(imageData),
    });
    if (!resp.ok) throw new Error('Failed to fetch processed image');
    return resp.blob();
  }

  static async genAIGuess(pngDataUrl, prompt) {
    const resp = await fetch(`${Config.BACKEND_URL}/genai_guess`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: pngDataUrl, prompt }),
    });
    if (!resp.ok) {
      const errorData = await resp.json().catch(() => ({}));
      throw new Error(errorData.detail || errorData.error || `Request failed with status ${resp.status}`);
    }
    return resp.json();
  }

  static async interpret(imageBase64, prediction, confidence) {
    const resp = await fetch(`${Config.BACKEND_URL}/interpret`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageBase64, prediction, confidence }),
    });
    if (!resp.ok) {
      const err = await resp.json().catch(() => ({}));
      throw new Error(err.error || err.detail || 'Interpretation failed');
    }
    return resp.json();
  }
}

export default PredictionClient;
