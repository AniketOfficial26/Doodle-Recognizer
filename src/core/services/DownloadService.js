import { Time } from '../utils/Time';

export class DownloadService {
  static downloadDataUrl(filename, dataUrl) {
    const link = document.createElement('a');
    link.download = filename;
    link.href = dataUrl;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  }

  static downloadBlob(filename, blob) {
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  static timestamped(base, ext = 'png') {
    return `${base}_${Time.timestamp()}.${ext}`;
  }
}

export default DownloadService;
