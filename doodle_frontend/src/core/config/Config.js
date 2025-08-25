// Centralized configuration for the frontend
export class Config {
  // Keep parity with existing behavior
  static BACKEND_URL =
    process.env.REACT_APP_API_URL ||
    process.env.REACT_APP_BACKEND_URL ||
    'http://localhost:5001';
}

export default Config;
