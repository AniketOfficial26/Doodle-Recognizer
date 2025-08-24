export class Time {
  static timestamp(date = new Date()) {
    return date
      .toISOString()
      .replace(/[:.]/g, '-')
      .replace('T', '_')
      .split('+')[0];
  }
}

export default Time;
