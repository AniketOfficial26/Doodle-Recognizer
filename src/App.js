/** @jsxImportSource @emotion/react */
import React, { useRef, useState, useEffect } from 'react';
import { css } from '@emotion/react';
import styled from '@emotion/styled';
import colors from './constants/colors';
import { HeaderBadges, Badge, Chip, ProgressWrap, ProgressBar } from './ui/Styled';
import PredictionResults from './ui/PredictionResults';
import AIAnalysisPanel from './ui/AIAnalysisPanel';
import InfoModal from './ui/InfoModal';
import TermsModal from './ui/TermsModal';
import FooterBar from './ui/FooterBar';
import { appShiny as shiny, appPulse as pulse, AppGlowingText as GlowingText, AppShinyText as ShinyText } from './ui/Styled';

// Centralized backend URL to avoid port mismatches
const BACKEND_URL = process.env.REACT_APP_BACKEND_URL || 'http://localhost:5001';

// Colors are imported from './constants/colors'

// Fixed canvas size - do not change
const getCanvasSize = () => {
  return { width: 800, height: 600 };
};

// Gamified UI elements and heading texts are now imported from './ui/Styled'

const DEFAULT_STROKE_COLOR = '#000000'; // default black stroke for drawing

// Component styles
const AppContainer = styled.div`
  height: 100vh;
  max-height: 100vh;
  overflow: hidden;
  background: radial-gradient(1200px 600px at 20% -10%, rgba(99, 102, 241, 0.25), transparent 50%),
              radial-gradient(1000px 500px at 120% 10%, rgba(244, 114, 182, 0.2), transparent 40%),
              ${colors.background};
  color: ${colors.text};
  display: flex;
  flex-direction: column;
  position: relative;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
  
  /* Allow page scroll on mobile so header/footer can be revealed */
  @media (max-width: 768px) {
    /* Allow page scroll on mobile */
    height: auto;
    max-height: none;
    overflow: auto;
  }
`;

const MainContainer = styled.div`
  flex: 1;
  padding: 8px 12px;
  max-width: 1600px;
  margin: 0 auto;
  width: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  
  @media (max-width: 768px) {
    overflow: visible;
  }
`;

// Decorative side doodles to fill side whitespace on large screens
const SideDoodle = styled.div`
  position: absolute;
  top: 50%;
  ${props => props.side === 'left' ? 'left: -80px;' : 'right: -80px;'}
  transform: translateY(-50%) rotate(${props => props.side === 'left' ? '-10deg' : '10deg'});
  opacity: 0.16;
  pointer-events: none;
  z-index: 0;
  filter: drop-shadow(0 8px 18px rgba(0,0,0,0.25));

  svg { width: 360px; height: auto; }

  @media (max-width: 1200px) { display: none; }
`;

const Header = styled.div`
  background: linear-gradient(180deg, rgba(30, 41, 59, 0.6), rgba(30, 41, 59, 0.35));
  backdrop-filter: blur(14px);
  border-radius: 16px;
  padding: 10px 12px;
  margin-bottom: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 8px 30px rgba(0,0,0,0.35);
  border: 1px solid rgba(99, 102, 241, 0.35);
  position: relative;
  overflow: hidden;

  &::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #ff8a00, #e52e71, #0073e6, #00b4ff, #ff8a00);
    background-size: 200% 100%;
    animation: ${shiny} 8s linear infinite;
  }
  
  @media (max-width: 768px) {
    padding: 8px 10px;
    border-radius: 12px;
  }
`;

const HeaderLeft = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
  flex: 1; /* allow left side (icon + title) to take available space */

  @media (max-width: 768px) {
    gap: 10px;
  }
`;

const Footer = styled.footer`
  position: relative;
  bottom: 0;
  left: 0;
  margin: 4px 0 0;
  padding: 1px 0;
  width: 100%;
  background: rgba(0, 0, 0, 0.95);
  text-align: center;
  color: #94a3b8;
  font-size: 0.5rem;
  line-height: 1.2;
  border-top: 1px solid rgba(99, 102, 241, 0.2);
  z-index: 10;

  &::before {
    content: "";
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 2px;
    background: linear-gradient(90deg, #ff8a00, #e52e71, #0073e6, #00b4ff, #ff8a00);
    background-size: 200% 100%;
    animation: ${shiny} 8s linear infinite;
  }
  
  a {
    color: #a5b4fc;
    margin: 0 3px;
    text-decoration: none;
    font-size: 0.58rem;
    padding: 0 3px;

    &:hover {
      color: #6366f1;
      background: rgba(99, 102, 241, 0.1);
      text-decoration: none;
    }
  }
`;

const BrushIcon = styled.div`
  color: ${colors.primary};
  font-size: 32px;
  filter: drop-shadow(0 0 8px rgba(99, 102, 241, 0.7));
  animation: ${pulse} 2s infinite;

  &::before {
    content: "🖌️";
  }
`;

const InfoButton = styled.button`
  color: #fff;
  background: rgba(99, 102, 241, 0.2);
  backdrop-filter: blur(5px);
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 8px;
  padding: 8px;
  cursor: pointer;
  transition: all 0.3s ease;
  animation: ${pulse} 2s infinite;

  &:hover {
    background: rgba(99, 102, 241, 0.4);
    box-shadow: 0 0 15px rgba(99, 102, 241, 0.4);
  }

  &::before {
    content: "ℹ️";
    font-size: 16px;
  }
`;

const ColorPicker = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0;
  padding: 8px;
  background: ${colors.surface2};
  border-radius: 8px;
`;

const MainContent = styled.div`
  display: flex;
  flex: 1;
  gap: 6px;
  min-height: 0;
  /* Allow natural height so bottom buttons are not clipped */
  max-height: none;
  overflow: visible;

  & > *:first-child {
    width: 250px;
    flex-shrink: 0;
    overflow-y: visible;
  }
  
  & > *:last-child {
    width: 250px;
    flex-shrink: 0;
    overflow-y: auto;
  }

  & > *:nth-child(2) {
    flex: 1;
    display: flex;
    justify-content: center;
    align-items: center;
    overflow: hidden;
  }

  /* Mobile: hide side panels and let canvas take full space */
  @media (max-width: 768px) {
    display: block;
    gap: 0;
    max-height: none;
    & > *:first-child,
    & > *:last-child { display: none; }
    & > *:nth-child(2) { flex: 1; }
  }

  &::-webkit-scrollbar {
    width: 4px;
  }

  &::-webkit-scrollbar-track {
    background: #f1f1f1;
    border-radius: 2px;
  }

  &::-webkit-scrollbar-thumb {
    background: #90caf9;
    border-radius: 2px;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: #64b5f6;
  }
`;

const Panel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
  overflow-y: visible;
  height: calc(100vh - 150px);
  min-width: 250px;
  background: linear-gradient(180deg, rgba(99, 102, 241, 0.16), rgba(99, 102, 241, 0.08));
  backdrop-filter: blur(12px);
  border-radius: 20px;
  padding: 16px;
  border: 1px solid rgba(99, 102, 241, 0.28);
  box-shadow: 0 10px 30px rgba(0,0,0,0.25), inset 0 1px 0 rgba(255,255,255,0.06);
  transition: all 0.3s ease;
  
  &:hover {
    box-shadow: 0 14px 40px rgba(99, 102, 241, 0.22);
    border-color: rgba(99, 102, 241, 0.38);
  }
  
  /* Hide panels on mobile */
  @media (max-width: 768px) {
    display: none;
  }
`;

const DarkPanel = styled.div`
  background: linear-gradient(180deg, rgba(30, 41, 59, 0.65), rgba(30, 41, 59, 0.35));
  backdrop-filter: blur(12px);
  box-shadow: 0 8px 30px rgba(0,0,0,0.32);
  border: 1px solid rgba(99, 102, 241, 0.3);
  border-radius: 16px;
  padding: 16px;
  margin-bottom: 16px;
  transition: all 0.3s ease;
  
  &:hover {
    background: linear-gradient(180deg, rgba(30, 41, 59, 0.72), rgba(30, 41, 59, 0.42));
    box-shadow: 0 12px 36px rgba(99, 102, 241, 0.28);
  }
`;

const SectionTitle = styled.h3`
  margin-bottom: 12px;
  font-weight: 800;
  font-size: 0.95rem;
  letter-spacing: 0.6px;
  text-transform: uppercase;
  margin: 0 0 12px 0;
  background: linear-gradient(90deg, ${colors.accentBlue}, ${colors.accentPurple}, ${colors.accentPink});
  background-size: 200% auto;
  -webkit-background-clip: text;
  background-clip: text;
  -webkit-text-fill-color: transparent;
`;

const Button = styled.button`
  background: ${props => props.primary 
    ? 'linear-gradient(135deg, #7c3aed 0%, #4f46e5 50%, #06b6d4 100%)' 
    : 'rgba(255,255,255,0.06)'};
  color: ${props => props.primary ? '#ffffff' : colors.text};
  border: ${props => props.primary ? '1px solid rgba(255,255,255,0.12)' : `1px solid rgba(148, 163, 184, 0.25)`};
  border-radius: 9999px;
  padding: 10px 18px;
  cursor: pointer;
  font-size: 0.8rem;
  font-weight: 700;
  text-transform: none;
  transition: all 0.2s ease;
  width: ${props => props.fullWidth ? '100%' : 'auto'};
  margin-bottom: 8px;
  min-height: 40px;
  box-shadow: ${props => props.primary ? '0 8px 20px rgba(79, 70, 229, 0.35)' : '0 4px 12px rgba(0,0,0,0.25)'};

  &:hover {
    filter: brightness(1.06);
    transform: translateY(-1px);
    box-shadow: ${props => props.primary ? '0 12px 26px rgba(79, 70, 229, 0.45)' : '0 6px 16px rgba(0,0,0,0.3)'};
  }

  &:active {
    transform: translateY(0);
  }

  &:disabled {
    opacity: 0.6;
    cursor: not-allowed;
    transform: none;
  }
`;

const CanvasContainer = styled.div`
  flex: 1;
  background: ${colors.surface};
  border-radius: 12px;
  padding: 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  border: 1px solid ${colors.surface2};
  padding: 16px;
  box-sizing: border-box;
  position: relative;
  
  /* Full-screen canvas on mobile */
  @media (max-width: 768px) {
    border-radius: 0;
    border: none;
    padding: 0;
    background: ${colors.background};
  }
`;

const CanvasWrapper = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 16px;
  box-sizing: border-box;
  position: relative;
  /* Reserve space for the bottom ButtonGroup on desktop to avoid clipping */
  @media (min-width: 769px) {
    max-height: calc(100% - 72px);
  }
  
  @media (max-width: 768px) {
    padding: 0;
    width: 100%;
    height: auto; /* allow content to determine height for scrolling */
  }
`;

const Canvas = styled.canvas`
  background: ${colors.surface2};
  border-radius: 8px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  cursor: crosshair;
  border: 1px solid ${colors.surface2};
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  transition: all 0.3s ease;

  &:hover {
    border-color: rgba(99, 102, 241, 0.5);
    box-shadow: 0 6px 35px rgba(99, 102, 241, 0.15);
  }
  
  @media (max-width: 768px) {
    border-radius: 12px;
    border: 2px solid rgba(255,255,255,0.35);
    width: 100%;
    height: 100%;
    background: #ffffff; /* white canvas on mobile */
  }
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 16px;
  margin-top: 16px;
  width: 100%;
  max-width: ${props => props.maxWidth || '100%'};
  justify-content: center;
  padding: 0 20px;
  min-height: 56px;

  & > * {
    flex: 1;
    max-width: 220px;
    margin-bottom: 0; /* prevent partial clipping */
  }
  
  /* Hide desktop-style action buttons on mobile (we have a sticky bar) */
  @media (max-width: 768px) {
    display: none;
  }
`;

/* Framed container behind the canvas to match mobile screenshot */
const CanvasFrame = styled.div`
  display: inline-block;
  padding: 0; /* no effect on desktop */
  border-radius: 0;
  background: transparent;
  border: none;

  @media (max-width: 768px) {
    padding: 12px;
    border-radius: 18px;
    background: linear-gradient(180deg, rgba(46, 64, 117, 0.9), rgba(46, 64, 117, 0.7));
    box-shadow: inset 0 2px 0 rgba(255,255,255,0.15), 0 6px 24px rgba(0,0,0,0.35);
    border: 1px solid rgba(255,255,255,0.12);
    width: 100%;
    height: auto; /* let content flow */
    display: flex;
    align-items: center;
    justify-content: center;
  }
`;

/* Mobile hamburger button */
const HamburgerButton = styled.button`
  display: none;
  @media (max-width: 768px) {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 36px;
    height: 36px;
    margin-right: 8px;
    border-radius: 8px;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    color: #fff;
    cursor: pointer;
  }
`;

/* Mobile menu overlay */
const MobileMenuOverlay = styled.div`
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.8);
  backdrop-filter: blur(6px);
  z-index: 1100;
  display: flex;
  align-items: flex-end;
  @media (min-width: 769px) { display: none; }
`;

const MobileMenuContent = styled.div`
  width: 100%;
  max-height: 85vh;
  background: ${colors.surface};
  border-top-left-radius: 16px;
  border-top-right-radius: 16px;
  border: 1px solid ${colors.surface2};
  padding: 16px;
  overflow-y: auto;
`;

const MobileMenuHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  color: ${colors.text};
`;

const MobilePredictionBox = styled.div`
  background: ${colors.surface2};
  border: 1px solid ${colors.surface2};
  border-radius: 12px;
  padding: 12px;
  margin-top: 12px;
  color: ${colors.text};
  font-size: 14px;
`;

/* Wrapper to show only on mobile */
const MobileOnly = styled.div`
  display: none;
  @media (max-width: 768px) { display: block; }
`;

/* Inline actions bar for mobile (below canvas) */
const MobileActionsBar = styled.div`
  display: none;
  @media (max-width: 768px) {
    display: flex;
    position: fixed;
    left: 0;
    right: 0;
    bottom: 0;
    /* Ensure full button visibility with adequate height and safe-area */
    min-height: 92px;
    padding: 12px 12px calc(env(safe-area-inset-bottom) + 14px);
    background: rgba(15, 23, 42, 0.9);
    backdrop-filter: blur(8px);
    gap: 10px;
    justify-content: center;
    width: 100%;
    z-index: 1200;
    border-top: 1px solid rgba(99, 102, 241, 0.25);
    box-shadow: 0 -6px 18px rgba(0,0,0,0.35);

    /* Prevent child buttons from adding bottom margin that causes clipping */
    & > * { 
      margin-bottom: 0 !important; 
      min-height: 44px; 
      flex: 1 1 0;
    }
  }
`;

const Modal = styled.div`
  position: fixed;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  justify-content: center;
  align-items: center;
  z-index: 1000;
  display: ${props => props.open ? 'flex' : 'none'};
  align-items: center;
`;

const ModalContent = styled.div`
  background: ${colors.surface};
  padding: 32px;
  border-radius: 12px;
  border: 1px solid ${colors.surface2};
  max-width: 90%;
  max-height: 90%;
  overflow-y: auto;
  position: relative;
  box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(99, 102, 241, 0.3);
  border: 1px solid rgba(99, 102, 241, 0.2);
box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(99, 102, 241, 0.3);
border: 1px solid rgba(99, 102, 241, 0.2);
  box-shadow: 0 20px 60px rgba(0, 0, 0, 0.6), 0 0 0 1px rgba(99, 102, 241, 0.3);
  border: 1px solid rgba(99, 102, 241, 0.2);

  &::-webkit-scrollbar {
    width: 8px;
  }

  &::-webkit-scrollbar-track {
    background: rgba(255,255,255,0.1);
    border-radius: 4px;
  }

  &::-webkit-scrollbar-thumb {
    background: rgba(99, 102, 241, 0.6);
    border-radius: 4px;
  }

  &::-webkit-scrollbar-thumb:hover {
    background: rgba(99, 102, 241, 0.8);
  }
`;

const CloseButton = styled.button`
  position: absolute;
  top: 16px;
  right: 16px;
  color: #a5b4fc;
  background: rgba(255,255,255,0.1);
  border: none;
  border-radius: 8px;
  padding: 8px;
  cursor: pointer;
  transition: all 0.2s ease;

  &:hover {
    background: rgba(255,255,255,0.2);
    transform: scale(1.1);
  }

  &::before {
    content: "✕";
    font-size: 16px;
  }
`;

const LoadingSpinner = styled.div`
  border: 2px solid #f3f3f3;
  border-top: 2px solid ${colors.secondary};
  border-radius: 50%;
  width: 24px;
  height: 24px;
  animation: spin 1s linear infinite;
  margin-bottom: 8px;

  @keyframes spin {
    0% { transform: rotate(0deg); }
    100% { transform: rotate(360deg); }
  }
`;

const FeedbackBox = styled.div`
  position: fixed;
  top: 20px;
  right: 20px;
  background-color: rgba(99, 102, 241, 0.9);
  color: white;
  padding: 8px 24px;
  border-radius: 8px;
  z-index: 1000;
  box-shadow: 0 4px 12px rgba(0,0,0,0.3);
`;

function App() {
  // Canvas and drawing state
  const canvasRef = useRef(null);
  const [drawing, setDrawing] = useState(false);
  const [prediction, setPrediction] = useState("");
  const [predictionConfidence, setPredictionConfidence] = useState(0);
  const [topPredictions, setTopPredictions] = useState([]);
  const [genAiPrediction, setGenAiPrediction] = useState("");
  const [genAiLoading, setGenAiLoading] = useState(false);
  const [loading, setLoading] = useState(false);
  const [showTerms, setShowTerms] = useState(false);
  const [activeTab, setActiveTab] = useState('privacy'); // 'privacy' or 'terms'
  // Gamified UI state (UI-only; does not affect app logic)
  const [streak, setStreak] = useState(0);
  const [xp, setXp] = useState(0);
  
  // UI state
  const [feedback, setFeedback] = useState('');
  const [aiImage, setAiImage] = useState(""); // base64 or URL for AI image
  const [isErasing, setIsErasing] = useState(false); // new state for eraser
  const [isEnhancing, setIsEnhancing] = useState(false); // Loading state for enhance button
  const [showInfo, setShowInfo] = useState(false);
  const [strokeWidth, setStrokeWidth] = useState(36); // Increased default brush size to 36
  const [brushColor, setBrushColor] = useState(DEFAULT_STROKE_COLOR);
  const [canvasSize, setCanvasSize] = useState(getCanvasSize());
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  // Remove responsive canvas size handler since canvas size is now fixed
  // useEffect(() => {
  //   const handleResize = () => {
  //     setCanvasSize(getCanvasSize());
  //   };
  //   window.addEventListener('resize', handleResize);
  //   return () => window.removeEventListener('resize', handleResize);
  // }, []);

  // Info modal keyboard handler
  useEffect(() => {
    const handleKeyDown = (event) => {
      if (event.key === 'Escape' && showInfo) {
        setShowInfo(false);
      }
    };
    
    if (showInfo) {
      document.addEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'hidden';
    } else {
      document.body.style.overflow = 'unset';
    }
    
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      document.body.style.overflow = 'unset';
    };
  }, [showInfo]);

  // Initialize canvas
  useEffect(() => {
    if (canvasRef.current) {
      clearCanvas();
    }
  }, [canvasSize]);

  // Initial reset
  useEffect(() => {
    resetCanvasAndFeedback();
  }, []);

  // Generate AI image via Stability AI and download it
  const handleEnhance = async () => {
    if (!canvasRef.current) return;
    try {
      setIsEnhancing(true);

      // Convert canvas to data URL (PNG)
      const dataUrl = canvasRef.current.toDataURL('image/png');
      if (!dataUrl || !dataUrl.startsWith('data:image/')) {
        throw new Error('Failed to capture canvas image');
      }

      // Build request payload per StabilityGenerateRequest schema
      const payload = {
        image: dataUrl,
        // Encourage AI to reinterpret the sketch, not copy the rough strokes
        prompt: prediction
          ? `Clean, high-quality ${prediction} recreated from the sketch; preserve composition and silhouette, correct proportions, smooth refined lines, do not replicate the rough sketch strokes, high quality.`
          : `Clean, high-quality subject recreated from the sketch; preserve composition and silhouette, correct proportions, smooth refined lines, do not replicate the rough sketch strokes, high quality.`,
        strength: 0.2,
        output_format: 'png',
      };

      const resp = await fetch(`${BACKEND_URL}/stability_generate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (!resp.ok) {
        // Try to extract detailed error from backend
        let errDetail = '';
        try {
          const errJson = await resp.json();
          errDetail = errJson?.detail || errJson?.message || JSON.stringify(errJson);
        } catch (_) {
          try {
            errDetail = await resp.text();
          } catch (_) { /* ignore */ }
        }
        throw new Error(errDetail || `Request failed (${resp.status})`);
      }

      const data = await resp.json();
      const { image_base64, format } = data || {};
      if (!image_base64) {
        throw new Error('No image returned from Stability AI');
      }

      // Create data URL for display and download
      const mime = `image/${(format || 'png').toLowerCase()}`;
      const url = `data:${mime};base64,${image_base64}`;

      // Set the AI image for display
      setAiImage(url);

      // Auto-download
      const a = document.createElement('a');
      a.href = url;
      a.download = `generated_doodle_${(prediction || 'image').replace(/\s+/g,'_')}_${new Date().toISOString().replace(/[:.]/g,'-')}.${(format || 'png').toLowerCase()}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (e) {
      console.error('Stability generate error:', e);
      setGenAiPrediction(`Error generating AI image: ${e?.message || e}`);
    } finally {
      setIsEnhancing(false);
    }
  };

  // Download user drawing in original size as PNG
  const downloadUserDrawing = () => {
    const canvas = canvasRef.current;
    
    // Create a temporary canvas with the same size as the original
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Fill with white background
    tempCtx.fillStyle = '#FFFFFF';
    tempCtx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Draw the original content
    tempCtx.drawImage(canvas, 0, 0);
    
    // Convert to PNG with no compression
    const pngData = tempCanvas.toDataURL('image/png');
    
    // Get current date and time for filename
    const now = new Date();
    const timestamp = now.toISOString()
      .replace(/[:.]/g, '-')
      .replace('T', '_')
      .split('+')[0];
    
    // Download the image
    const link = document.createElement('a');
    link.download = `doodle_${timestamp}.png`;
    link.href = pngData;
    link.click();
  };

  const downloadAIDrawing = () => {
    if (!aiImage) return;
    
    // Get current date and time for filename
    const now = new Date();
    const timestamp = now.toISOString()
      .replace(/[:.]/g, '-')
      .replace('T', '_')
      .split('+')[0];
    
    // Create a temporary anchor element to trigger download
    const link = document.createElement('a');
    link.download = `ai_doodle_${prediction || 'generated'}_${timestamp}.png`;
    link.href = aiImage;
    
    // Add to document, trigger click, and clean up
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    // Show feedback to user
    setFeedback('AI doodle downloaded!');
    setTimeout(() => setFeedback(''), 2000);
  };

  const downloadProcessedImage = async () => {
    try {
      const imageData = getImageData();

      // Prevent blank downloads if nothing is drawn
      const hasDrawing = imageData.image.some((pixel) => pixel > 0.1);
      if (!hasDrawing) {
        setFeedback('Please draw something first');
        setTimeout(() => setFeedback(''), 1500);
        return;
      }

      const response = await fetch(`${BACKEND_URL}/download_processed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(imageData)
      });
      if (!response.ok) throw new Error('Failed to fetch processed image');
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'processed.png';
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      URL.revokeObjectURL(url);
      setFeedback('Processed image downloaded');
      setTimeout(() => setFeedback(''), 1500);
    } catch (err) {
      console.error('Processed image download failed:', err);
      setFeedback('Processed image download failed');
      setTimeout(() => setFeedback(''), 2000);
    }
  };

  const resetCanvasAndFeedback = () => {
    setFeedback('');
    setPrediction("");
    setPredictionConfidence(0);
    setTopPredictions([]);
    setGenAiPrediction("");
    clearCanvas();
  };

  // Drawing logic
  const getPointerPos = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY
    };
  };

  const handlePointerDown = (e) => {
    setDrawing(true);
    const pos = getPointerPos(e);
    const ctx = canvasRef.current.getContext('2d');
    ctx.beginPath();
    ctx.moveTo(pos.x, pos.y);
  };

  const handlePointerMove = (e) => {
    if (!drawing) return;
    const pos = getPointerPos(e);
    const ctx = canvasRef.current.getContext('2d');
    ctx.lineWidth = strokeWidth;
    ctx.lineCap = 'round';
    ctx.strokeStyle = isErasing ? '#ffffff' : brushColor;
    ctx.globalCompositeOperation = isErasing ? 'destination-out' : 'source-over';
    ctx.lineTo(pos.x, pos.y);
    ctx.stroke();
  };

  const handlePointerUp = () => {
    setDrawing(false);
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
  };

  const getImageData = () => {
    const canvas = canvasRef.current;
  
    // Create a 28x28 resized version for consistent ML input
    const tempCanvas = document.createElement('canvas');
    const tempCtx = tempCanvas.getContext('2d');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
  
    // Fill with white background
    tempCtx.fillStyle = '#FFFFFF';
    tempCtx.fillRect(0, 0, 28, 28);
  
    // Draw the original canvas scaled to 28x28
    tempCtx.drawImage(canvas, 0, 0, canvas.width, canvas.height, 0, 0, 28, 28);
  
    // Get image data from the resized canvas
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;
  
    // Convert to grayscale and normalize to 0-1 range
    const normalizedData = [];
    for (let i = 0; i < data.length; i += 4) {
      const r = data[i];
      const g = data[i + 1];
      const b = data[i + 2];
      
      // Calculate grayscale value (0-255)
      const gray = (r + g + b) / 3;
      
      // Invert and normalize: 
      // - White pixels (255) become 0 (background)
      // - Black pixels (0) become 1 (drawing)
      // - Gray pixels are scaled accordingly
      const normalized = (255 - gray) / 255;
      normalizedData.push(normalized);
    }
  
    return {
      image: normalizedData,
      width: 28,
      height: 28
    };
  };

  const getAIInterpretation = async (imageData, prediction, confidence) => {
    try {
      setGenAiLoading(true);
      // Get the canvas as base64 image
      const canvas = canvasRef.current;
      const imageBase64 = canvas.toDataURL('image/jpeg', 0.8);
      
      const response = await fetch(`${BACKEND_URL}/interpret`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          image: imageBase64,
          prediction: prediction,
          confidence: confidence
        }),
      });
      
      const result = await response.json();
      if (result.error) {
        throw new Error(result.error);
      }
      
      return result.interpretation;
    } catch (err) {
      console.error("AI interpretation error:", err);
      return "I'm having trouble analyzing this drawing right now. Please try again later.";
    } finally {
      setGenAiLoading(false);
    }
  };

  const announcePrediction = (message) => {
    if (!message) return;
    
    // Cancel any ongoing speech
    window.speechSynthesis.cancel();
    
    // Create a new speech synthesis utterance
    const utterance = new SpeechSynthesisUtterance(message);
    
    // Set voice properties
    utterance.rate = 0.9; // Slightly slower than normal
    utterance.pitch = 1.1; // Slightly higher pitch
    utterance.volume = 1.0; // Full volume
    
    // Find a good voice
    const voices = window.speechSynthesis.getVoices();
    const preferredVoices = voices.filter(voice => 
      voice.lang.includes('en') && voice.name.includes('Google')
    );
    
    if (preferredVoices.length > 0) {
      utterance.voice = preferredVoices[0];
    } else if (voices.length > 0) {
      // Fallback to any available English voice
      const englishVoices = voices.filter(voice => voice.lang.includes('en'));
      if (englishVoices.length > 0) {
        utterance.voice = englishVoices[0];
      }
    }
    
    // Speak the message
    window.speechSynthesis.speak(utterance);
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const imageData = getImageData();

      // Check if there's any drawing on the canvas
      const hasDrawing = imageData.image.some((pixel) => pixel > 0.1);
      if (!hasDrawing) {
        const message = "Please draw something on the canvas first.";
        setPrediction("No Drawing");
        setPredictionConfidence(0);
        setTopPredictions([]);
        announcePrediction(message);
        setLoading(false);
        return;
      }

      // Speak that we're processing (optional UI feedback only for model prediction)
      announcePrediction("Analyzing your drawing...");

      const response = await fetch(`${BACKEND_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(imageData),
      });

      if (!response.ok) {
        let errMsg = 'Prediction failed';
        try {
          const err = await response.json().catch(() => ({}));
          if (err && (err.detail || err.message)) {
            errMsg = err.detail || err.message;
          }
        } catch {}
        setPrediction("Error");
        setPredictionConfidence(0);
        setTopPredictions([]);
        announcePrediction(`${errMsg}.`);
        return;
      }

      const result = await response.json();

      if (result.error) {
        const errorMessage = result.error || "Something went wrong. Please try again.";
        setPrediction("Error");
        setPredictionConfidence(0);
        setTopPredictions([]);
        announcePrediction(errorMessage);
        return;
      }

      // Update the UI with prediction results
      setPrediction(result.label);
      setPredictionConfidence(result.confidence);
      // UI-only gamification counters
      setStreak((prev) => prev + 1);
      setXp((prev) => prev + 10);

      // Process top predictions if available
      let predictions = [];
      if (result.top_predictions && result.top_predictions.length > 0) {
        predictions = result.top_predictions;
      } else {
        predictions = [{ class: result.label, confidence: result.confidence }];
      }
      setTopPredictions(predictions);
      // Do not trigger or set AI analysis here; keep AI analysis independent
      const confPct = Math.round((result.confidence || 0) * 100);
      announcePrediction(`I think it's ${result.label}. Confidence ${confPct} percent.`);

      // Automatically trigger AI analysis after model prediction completes
      // This keeps AI analysis logically independent but improves UX by auto-running it.
      handleGenAiPredict();
    } catch (err) {
      console.error("Prediction error:", err);
      const errorMessage = "Error making prediction. Please try again.";
      setPrediction("Error");
      setPredictionConfidence(0);
      setTopPredictions([]);
      announcePrediction(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleGenAiPredict = async () => {
    try {
      setGenAiLoading(true);
      setAiImage(""); // Clear previous image
      const canvas = canvasRef.current;

      // Show a loading message
      setGenAiPrediction('Analyzing your drawing with AI...');

      // Call the genai_guess endpoint
      const response = await fetch(`${BACKEND_URL}/genai_guess`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          image: canvas.toDataURL('image/png'),
          prompt: `What is this drawing? Choose one of: apple, airplane, cat, car, dog, flower, star, tree, umbrella, fish.`
        })
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new Error(errorData.detail || errorData.error || `Request failed with status ${response.status}`);
      }

      const data = await response.json();

      if (data.guess) {
        // Update the prediction with the AI's guess
        setGenAiPrediction(`The AI thinks this is a: ${data.guess}`);
      } else {
        throw new Error('No guess returned from AI');
      }
    } catch (error) {
      console.error('Error getting AI guess:', error);
      setGenAiPrediction(`Error: ${error.message || 'Failed to get AI analysis. Please try again.'}`);
    } finally {
      setGenAiLoading(false);
    }
  };

  return (
    <AppContainer>
      {/* Decorative side doodles (non-interactive) */}
      <SideDoodle side="left" aria-hidden>
        <svg viewBox="0 0 300 600" fill="none" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="doodleGradL" x1="0" y1="0" x2="300" y2="0" gradientUnits="userSpaceOnUse">
              <stop stopColor="#60a5fa"/>
              <stop offset="0.5" stopColor="#a78bfa"/>
              <stop offset="1" stopColor="#f472b6"/>
            </linearGradient>
          </defs>
          <path d="M10 20 C 100 60, 40 140, 120 180 S 220 280, 160 340 S 60 450, 140 520" stroke="url(#doodleGradL)" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round"/>
          <circle cx="80" cy="100" r="10" fill="#60a5fa"/>
          <circle cx="200" cy="260" r="8" fill="#a78bfa"/>
          <circle cx="120" cy="420" r="6" fill="#f472b6"/>
        </svg>
      </SideDoodle>
      <SideDoodle side="right" aria-hidden>
        <svg viewBox="0 0 300 600" fill="none" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <linearGradient id="doodleGradR" x1="300" y1="0" x2="0" y2="0" gradientUnits="userSpaceOnUse">
              <stop stopColor="#f472b6"/>
              <stop offset="0.5" stopColor="#a78bfa"/>
              <stop offset="1" stopColor="#60a5fa"/>
            </linearGradient>
          </defs>
          <path d="M290 40 C 220 100, 260 160, 200 210 S 100 310, 180 370 S 260 470, 190 520" stroke="url(#doodleGradR)" strokeWidth="8" strokeLinecap="round" strokeLinejoin="round"/>
          <circle cx="220" cy="140" r="9" fill="#f472b6"/>
          <circle cx="120" cy="330" r="7" fill="#a78bfa"/>
          <circle cx="210" cy="480" r="6" fill="#60a5fa"/>
        </svg>
      </SideDoodle>
      <MainContainer>
        {/* Header */}
        <Header>
          <HeaderLeft>
            <BrushIcon />
            <GlowingText>
              <ShinyText>Doodle Recognizer</ShinyText>
            </GlowingText>
          </HeaderLeft>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <HamburgerButton aria-label="Open menu" onClick={() => setMobileMenuOpen(true)}>☰</HamburgerButton>
            <InfoButton onClick={() => setShowInfo(true)} />
          </div>
        </Header>
        {mobileMenuOpen && (
          <MobileMenuOverlay onClick={() => setMobileMenuOpen(false)}>
            <MobileMenuContent onClick={(e) => e.stopPropagation()}>
              <MobileMenuHeader>
                <span>Controls</span>
                <CloseButton onClick={() => setMobileMenuOpen(false)} />
              </MobileMenuHeader>

              <SectionTitle>Brush Size</SectionTitle>
              <ButtonGroup style={{ marginTop: 8 }}>
                <Button onClick={() => setStrokeWidth(Math.max(1, strokeWidth - 2))}>-</Button>
                <div style={{ alignSelf: 'center', color: '#a5b4fc', minWidth: 56, textAlign: 'center' }}>{strokeWidth}px</div>
                <Button onClick={() => setStrokeWidth(Math.min(100, strokeWidth + 2))}>+</Button>
              </ButtonGroup>

              <SectionTitle>Brush Color</SectionTitle>
              <ColorPicker>
                {['#000000','#ff0000','#00ff00','#0000ff','#ffffff','#ffa500','#800080','#00ffff'].map((c) => (
                  <button
                    key={c}
                    onClick={() => setBrushColor(c)}
                    style={{ width: 24, height: 24, borderRadius: 6, border: c === brushColor ? '2px solid #fff' : '1px solid #555', background: c, cursor: 'pointer' }}
                    aria-label={`Set color ${c}`}
                  />
                ))}
                <button
                  onClick={() => setIsErasing(!isErasing)}
                  style={{ marginLeft: 8, padding: '4px 8px', borderRadius: 6, border: '1px solid #555', background: isErasing ? 'rgba(255,255,255,0.2)' : 'transparent', color: '#fff' }}
                >
                  {isErasing ? 'Eraser On' : 'Eraser Off'}
                </button>
              </ColorPicker>

              <SectionTitle>Downloads</SectionTitle>
              <Button fullWidth onClick={downloadUserDrawing}>💾 Save Original</Button>
              <Button fullWidth onClick={downloadProcessedImage}>📥 Save Processed</Button>
            </MobileMenuContent>
          </MobileMenuOverlay>
        )}
        
        {/* Main Content - Three Column Layout */}
        <MainContent>
          {/* Left Panel - Drawing Tools */}
          <Panel>
            <DarkPanel>
              {/* Tools Section */}
              <div style={{ 
                background: 'rgba(99, 102, 241, 0.2)',
                borderRadius: '12px',
                padding: '12px',
                marginBottom: '16px',
                border: '1px solid rgba(99, 102, 241, 0.3)'
              }}>
                <h3 style={{ 
                  color: '#e2e8f0',
                  margin: '0 0 12px 0',
                  fontWeight: '600',
                  fontSize: '0.95rem',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px'
                }}>
                  Drawing Tools
                </h3>
                
                <p style={{ 
                  color: '#a5b4fc', 
                  marginBottom: '12px', 
                  fontSize: '0.85rem',
                  fontWeight: '500',
                  margin: '0 0 12px 0'
                }}>
                  Brush Size
                </p>
                <div style={{ 
                  display: 'flex', 
                  alignItems: 'center', 
                  gap: '6px', 
                  marginBottom: '16px',
                  background: 'rgba(99, 102, 241, 0.2)',
                  borderRadius: '8px',
                  padding: '6px',
                  border: '1px solid rgba(99, 102, 241, 0.2)'
                }}>
                  <Button
                    onClick={() => setStrokeWidth(Math.max(1, strokeWidth - 2))}
                    style={{ 
                      minWidth: '30px', 
                      padding: '0', 
                      color: '#e2e8f0', 
                      borderColor: 'rgba(99, 102, 241, 0.4)',
                      minHeight: '30px',
                      margin: 0
                    }}
                  >-</Button>
                  <div style={{ 
                    background: 'rgba(99, 102, 241, 0.15)',
                    borderRadius: '4px',
                    padding: '4px 12px'
                  }}>
                    <span style={{ 
                      color: '#e2e8f0', 
                      minWidth: '40px', 
                      textAlign: 'center', 
                      fontSize: '0.85rem',
                      fontWeight: '600',
                      display: 'inline-block'
                    }}>
                      {strokeWidth}px
                    </span>
                  </div>
                  <Button
                    onClick={() => setStrokeWidth(Math.min(100, strokeWidth + 2))}
                    style={{ 
                      minWidth: '30px', 
                      padding: '0', 
                      color: '#e2e8f0', 
                      borderColor: 'rgba(99, 102, 241, 0.4)',
                      minHeight: '30px',
                      margin: 0
                    }}
                  >+</Button>
                </div>

                <Button
                  fullWidth
                  onClick={() => setIsErasing(!isErasing)}
                  style={{ 
                    marginBottom: '16px', 
                    textTransform: 'none', 
                    fontSize: '0.85rem', 
                    padding: '8px',
                    borderRadius: '8px',
                    borderWidth: '1px',
                    borderColor: isErasing ? 'rgba(239, 68, 68, 0.5)' : 'rgba(99, 102, 241, 0.4)',
                    backgroundColor: isErasing 
                      ? 'rgba(239, 68, 68, 0.1)' 
                      : 'rgba(99, 102, 241, 0.1)',
                    color: isErasing ? '#fecaca' : '#a5b4fc'
                  }}
                >
                  {isErasing ? '🧹 Eraser' : '✏️ Pencil'}
                </Button>

                <div style={{ marginBottom: '16px' }}>
                  <p style={{ 
                    color: '#a5b4fc', 
                    marginBottom: '12px', 
                    fontSize: '0.85rem',
                    fontWeight: '500',
                    margin: '0 0 12px 0'
                  }}>
                    Color
                  </p>
                  <div style={{ display: 'grid', gridTemplateColumns: 'repeat(6, 1fr)', gap: '2px', marginTop: '4px' }}>
                    {['#000000', '#dc2626', '#2563eb', '#10b981', '#f59e0b', '#8b5cf6'].map((color) => (
                      <div
                        key={color}
                        onClick={() => setBrushColor(color)}
                        style={{
                          width: '20px',
                          height: '20px',
                          borderRadius: '50%',
                          backgroundColor: color,
                          cursor: 'pointer',
                          border: brushColor === color ? '2px solid white' : '2px solid transparent',
                          transition: 'all 0.2s ease'
                        }}
                      />
                    ))}
                  </div>
                </div>
                
                <Button
                  fullWidth
                  onClick={resetCanvasAndFeedback}
                  style={{
                    color: '#e2e8f0',
                    borderColor: '#4b5563',
                    backgroundColor: 'rgba(75, 85, 99, 0.1)',
                    fontSize: '0.85rem',
                    fontWeight: '500',
                    textTransform: 'none',
                    borderRadius: '8px'
                  }}
                >
                  🗑️ Clear Canvas
                </Button>
              </div>
            </DarkPanel>

            {/* Actions Section */}
            <div style={{ 
              padding: '12px', 
              borderRadius: '8px',
              background: '#f8fafc',
              boxShadow: '0 2px 4px rgba(0,0,0,0.1)',
              marginBottom: '16px'
            }}>
              <h3 style={{ 
                marginBottom: '12px', 
                fontWeight: '800', 
                fontSize: '1.1rem',
                background: 'linear-gradient(90deg, #4a00e0, #8e2de2, #4a00e0)',
                backgroundSize: '200% auto',
                backgroundClip: 'text',
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                animation: `${shiny} 4s linear infinite`,
                margin: '0 0 12px 0'
              }}>
                ACTIONS
              </h3>
              
              <Button
                fullWidth
                primary
                onClick={handleEnhance}
                disabled={isEnhancing}
                style={{ textTransform: 'none', fontSize: '0.75rem' }}
              >
                {isEnhancing ? '⏳ Generating...' : '🎨 Generate AI Image'}
              </Button>

              <Button
                fullWidth
                onClick={downloadUserDrawing}
                style={{ 
                  textTransform: 'none', 
                  fontSize: '0.92rem',
                  fontWeight: '600',
                  backgroundColor: 'rgba(49, 46, 129, 0.95)',
                  color: '#ffffff',
                  textShadow: '0 1px 1px rgba(0, 0, 0, 0.36)',
                  border: '1px solid rgba(99, 102, 241, 0.8)',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                  marginBottom: '8px',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    backgroundColor: 'rgba(37, 35, 96, 0.95)',
                    transform: 'translateY(-1px)',
                    boxShadow: '0 4px 8px rgba(0,0,0,0.25)'
                  }
                }}
              >
                💾 Save Drawing
              </Button>

              <Button
                fullWidth
                onClick={downloadProcessedImage}
                style={{ 
                  textTransform: 'none', 
                  fontSize: '0.8rem',
                  fontWeight: '600',
                  backgroundColor: 'rgba(49, 46, 129, 0.95)',
                  color: '#ffffff',
                  textShadow: '0 1px 1px rgba(0,0,0,0.3)',
                  border: '1px solid rgba(99, 102, 241, 0.8)',
                  boxShadow: '0 2px 4px rgba(0,0,0,0.2)',
                  transition: 'all 0.2s ease',
                  '&:hover': {
                    backgroundColor: 'rgba(37, 35, 96, 0.95)',
                    transform: 'translateY(-1px)',
                    boxShadow: '0 4px 8px rgba(0,0,0,0.25)'
                  }
                }}
              >
                📥 Save Processed
              </Button>
            </div>
          </Panel>
          
          {/* Center Panel - Canvas */}
          <CanvasContainer>
            <CanvasWrapper>
              <CanvasFrame>
                <Canvas
                  ref={canvasRef}
                  width={canvasSize.width}
                  height={canvasSize.height}
                  onPointerDown={handlePointerDown}
                  onPointerMove={handlePointerMove}
                  onPointerUp={handlePointerUp}
                  onPointerLeave={handlePointerUp}
                  style={{
                    width: `${canvasSize.width}px`,
                    height: `${canvasSize.height}px`,
                  }}
                />
              </CanvasFrame>
              {/* Mobile-only action buttons below the canvas */}
              <MobileOnly>
                <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                  <Button
                    primary
                    onClick={handlePredict}
                    disabled={loading}
                    fullWidth
                    style={{ textTransform: 'none', fontSize: '0.9rem' }}
                  >
                    {loading ? 'Predicting…' : 'Predict Doodle'}
                  </Button>
                  <Button
                    onClick={handleEnhance}
                    disabled={isEnhancing}
                    fullWidth
                    style={{ textTransform: 'none', fontSize: '0.9rem' }}
                  >
                    {isEnhancing ? 'Generating…' : 'Generate AI'}
                  </Button>
                  <Button
                    onClick={clearCanvas}
                    fullWidth
                    style={{ textTransform: 'none', fontSize: '0.9rem' }}
                  >
                    Clear Canvas
                  </Button>
                </div>
              </MobileOnly>
              {/* Mobile-only prediction box below canvas */}
              <MobileOnly>
                <SectionTitle>Prediction</SectionTitle>
                <MobilePredictionBox>
                  {loading ? (
                    <div>Analyzing your drawing...</div>
                  ) : (
                    <>
                      <div style={{ marginBottom: 6 }}>
                        <strong>Result:</strong>
                        <div style={{ marginTop: 4 }}>
                          {prediction ? (
                            <>
                              <div style={{ fontSize: 16, fontWeight: 800, color: '#f8fafc' }}>{prediction}</div>
                              <div style={{ color: '#e2e8f0', fontWeight: 600, marginBottom: 6 }}>Confidence: {Math.round((predictionConfidence || 0) * 100)}%</div>
                              <ProgressWrap>
                                <ProgressBar value={Math.round((predictionConfidence || 0) * 100)} />
                              </ProgressWrap>
                            </>
                          ) : (
                            <div>No prediction yet</div>
                          )}
                        </div>
                      </div>
                      {topPredictions && topPredictions.length > 0 && (
                        <div style={{ marginTop: 6 }}>
                          <strong>Top guesses:</strong>
                          <ul style={{ marginTop: 4, paddingLeft: 18 }}>
                            {topPredictions.slice(0, 3).map((p, idx) => (
                              <li key={idx}>
                                {p.class} — {Math.round((p.confidence || 0) * 100)}%
                              </li>
                            ))}
                          </ul>
                        </div>
                      )}
                      {genAiPrediction && (
                        <div style={{ marginTop: 8 }}>
                          <strong>AI says:</strong>
                          <div style={{ marginTop: 4, whiteSpace: 'pre-wrap' }}>{genAiPrediction}</div>
                        </div>
                      )}
                    </>
                  )}
                </MobilePredictionBox>
              </MobileOnly>

              <ButtonGroup maxWidth={canvasSize.width + 'px'}>
                <Button
                  primary
                  onClick={handlePredict}
                  disabled={loading}
                  fullWidth
                  style={{ 
                    padding: '12px',
                    fontSize: '1rem',
                    fontWeight: '600',
                    textTransform: 'none',
                    borderRadius: '8px'
                  }}
                >
                  {loading ? 'Predicting...' : 'Predict Doodle'}
                </Button>
                <Button
                  onClick={handleEnhance}
                  disabled={isEnhancing}
                  fullWidth
                  style={{
                    color: '#e2e8f0',
                    borderColor: '#4b5563',
                    padding: '12px',
                    fontSize: '1rem',
                    fontWeight: '500',
                    textTransform: 'none',
                    borderRadius: '8px'
                  }}
                >
                  {isEnhancing ? 'Generating...' : 'Generate with AI'}
                </Button>
              </ButtonGroup>
            </CanvasWrapper>
          </CanvasContainer>
          
          {/* Right Panel - AI Analysis, Prediction Results & Actions */}
          <Panel>
            <AIAnalysisPanel genAiLoading={genAiLoading} genAiPrediction={genAiPrediction} />

            {/* Prediction Results Section */}
            <PredictionResults 
              prediction={prediction} 
              predictionConfidence={predictionConfidence} 
            />
          </Panel>
        </MainContent>

        

        {/* Feedback Display */}
        {feedback && (
          <FeedbackBox>
            <p style={{ margin: 0, fontSize: '0.875rem' }}>
              {feedback}
            </p>
          </FeedbackBox>
        )}
      </MainContainer>

      {/* Info Modal */}
      <InfoModal open={showInfo} onClose={() => setShowInfo(false)} />
      
      <FooterBar 
        onAbout={() => setShowInfo(true)}
        onPrivacy={() => { setActiveTab('privacy'); setShowTerms(true); }}
        onTerms={() => { setActiveTab('terms'); setShowTerms(true); }}
      />

      {/* Terms and Privacy Modal */}
      <TermsModal 
        open={showTerms}
        onClose={() => setShowTerms(false)}
        activeTab={activeTab}
        setActiveTab={setActiveTab}
      />
    </AppContainer>
  );
}

export default App; 