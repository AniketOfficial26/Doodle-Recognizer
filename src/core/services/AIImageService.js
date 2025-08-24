import Config from '../config/Config';

export class AIImageService {
  static async cartoonizeFromCanvas(canvas, label) {
    const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
    if (!blob) throw new Error('Failed to capture canvas');

    const form = new FormData();
    form.append('reference_image', blob, 'doodle.png');
    if (label && typeof label === 'string' && label.trim()) {
      form.append('label', label.trim());
    }

    const resp = await fetch(`${Config.BACKEND_URL}/cartoonize-doodle`, {
      method: 'POST',
      body: form,
    });
    if (!resp.ok) {
      let errText = '';
      try { errText = await resp.text(); } catch (_) {}
      throw new Error(errText || `Request failed (${resp.status})`);
    }
    return resp.blob();
  }
}

export default AIImageService;
