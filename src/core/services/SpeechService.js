export class SpeechService {
  static announce(message) {
    if (!message) return;
    const synth = window.speechSynthesis;
    try { synth.cancel(); } catch (_) {}

    const utterance = new SpeechSynthesisUtterance(message);
    utterance.rate = 0.9;
    utterance.pitch = 1.1;
    utterance.volume = 1.0;

    const voices = synth.getVoices();
    const preferred = voices.filter(v => v.lang?.includes('en') && v.name?.includes('Google'));
    if (preferred.length) utterance.voice = preferred[0];
    else {
      const english = voices.filter(v => v.lang?.includes('en'));
      if (english.length) utterance.voice = english[0];
    }

    synth.speak(utterance);
  }
}

export default SpeechService;
