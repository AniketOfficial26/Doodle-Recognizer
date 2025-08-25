/** @jsxImportSource @emotion/react */
import React, { useRef, useState, useEffect } from 'react';
import { css } from '@emotion/react';
import Config from './core/config/Config';
import { ImageProcessor } from './core/image/ImageProcessor';
import { PredictionClient } from './core/services/PredictionClient';
import { AIImageService } from './core/services/AIImageService';
import { SpeechService } from './core/services/SpeechService';
import { DownloadService } from './core/services/DownloadService';
import { Validation } from './core/utils/Validation';
import { Time } from './core/utils/Time';
import colors from './constants/colors';
import {
  GlowingText,
  ShinyText,
  AppContainer,
  MainContainer,
  Header,
  HeaderLeft,
  Footer,
  BrushIcon,
  InfoButton,
  ColorPicker,
  MainContent,
  Panel,
  DarkPanel,
  SectionTitle,
  Button,
  CanvasContainer,
  CanvasWrapper,
  Canvas,
  ButtonGroup,
  CanvasFrame,
  HamburgerButton,
  MobileMenuOverlay,
  MobileMenuContent,
  MobileMenuHeader,
  MobilePredictionBox,
  MobileOnly,
  MobileActionsBar,
  Modal,
  ModalContent,
  CloseButton,
  LoadingSpinner,
  FeedbackBox,
} from './ui/Styled';

// Centralized backend URL to avoid port mismatches
const BACKEND_URL = Config.BACKEND_URL;

// Color palette moved to src/constants/colors.js

// Fixed canvas size - do not change
const getCanvasSize = () => {
  return { width: 800, height: 600 };
};

const DEFAULT_STROKE_COLOR = '#000000'; // default black stroke for drawing

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

  // Generate AI image via Together AI and download it
  const handleEnhance = async () => {
    if (!canvasRef.current) return;
    try {
      setIsEnhancing(true);
      const outBlob = await AIImageService.cartoonizeFromCanvas(canvasRef.current, prediction);
      const url = URL.createObjectURL(outBlob);
      
      // Set the AI image for display
      setAiImage(url);
      
      // Auto-download
      const filename = DownloadService.timestamped('generated_doodle', 'png');
      // Use the same object URL for download to preserve behavior (no revoke)
      const a = document.createElement('a');
      a.href = url;
      a.download = filename;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
    } catch (e) {
      console.error('Generate error:', e);
      setGenAiPrediction(`Error generating AI image: ${e.message || e}`);
    } finally {
      setIsEnhancing(false);
    }
  };

  // Download user drawing in original size as PNG
  const downloadUserDrawing = () => {
    const canvas = canvasRef.current;
    // Create a white-background PNG from the original canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvas.width;
    tempCanvas.height = canvas.height;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.fillStyle = '#FFFFFF';
    tempCtx.fillRect(0, 0, canvas.width, canvas.height);
    tempCtx.drawImage(canvas, 0, 0);
    const pngData = tempCanvas.toDataURL('image/png');
    DownloadService.downloadDataUrl(DownloadService.timestamped('doodle', 'png'), pngData);
  };

  const downloadAIDrawing = () => {
    if (!aiImage) return;
    const filename = `ai_doodle_${(prediction || 'generated')}_${Time.timestamp()}.png`;
    DownloadService.downloadDataUrl(filename, aiImage);
    // Show feedback to user
    setFeedback('AI doodle downloaded!');
    setTimeout(() => setFeedback(''), 2000);
  };

  const downloadProcessedImage = async () => {
    try {
      const imageData = getImageData();
      if (!Validation.hasDrawing(imageData.image)) {
        setFeedback('Please draw something first');
        setTimeout(() => setFeedback(''), 1500);
        return;
      }
      const blob = await PredictionClient.downloadProcessed(imageData);
      DownloadService.downloadBlob('processed.png', blob);
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
    return ImageProcessor.to28x28Grayscale(canvasRef.current);
  };

  const getAIInterpretation = async (imageData, prediction, confidence) => {
    try {
      setGenAiLoading(true);
      const imageBase64 = canvasRef.current.toDataURL('image/jpeg', 0.8);
      const result = await PredictionClient.interpret(imageBase64, prediction, confidence);
      if (result.error) throw new Error(result.error);
      return result.interpretation;
    } catch (err) {
      console.error("AI interpretation error:", err);
      return "I'm having trouble analyzing this drawing right now. Please try again later.";
    } finally {
      setGenAiLoading(false);
    }
  };

  const announcePrediction = (message) => {
    SpeechService.announce(message);
  };

  const handlePredict = async () => {
    setLoading(true);
    try {
      const imageData = getImageData();
      
      // Check if there's any drawing on the canvas
      const hasDrawing = Validation.hasDrawing(imageData.image);
      if (!hasDrawing) {
        const message = "Please draw something on the canvas first.";
        setPrediction("No Drawing");
        setPredictionConfidence(0);
        setTopPredictions([]);
        setGenAiPrediction(message);
        announcePrediction(message);
        setLoading(false);
        return;
      }
      
      // Speak that we're processing
      announcePrediction("Analyzing your drawing...");
      const result = await PredictionClient.predict(imageData);
      
      if (result.error) {
        const errorMessage = result.error || "Something went wrong. Please try again.";
        setPrediction("Error");
        setPredictionConfidence(0);
        setTopPredictions([]);
        setGenAiPrediction(errorMessage);
        announcePrediction(errorMessage);
        return;
      }
      
      // Update the UI with prediction results
      setPrediction(result.label);
      setPredictionConfidence(result.confidence);
      
      // Process top predictions if available
      let predictions = [];
      if (result.top_predictions && result.top_predictions.length > 0) {
        predictions = result.top_predictions;
      } else {
        predictions = [{class: result.label, confidence: result.confidence}];
      }
      setTopPredictions(predictions);
      
      // Get AI interpretation if confidence is low or if it's the top prediction
      if (result.confidence < 0.7) {
        const loadingMessage = "Analyzing your drawing...";
        setGenAiPrediction(loadingMessage);
        announcePrediction(loadingMessage);
        
        const interpretation = await getAIInterpretation(
          imageData.image,
          result.label,
          result.confidence
        );
        
        setGenAiPrediction(interpretation);
        announcePrediction(interpretation);
      } else {
        const message = `I'm ${Math.round(result.confidence * 100)}% confident this is a ${result.label}!`;
        setGenAiPrediction(message);
        announcePrediction(message);
      }
      
    } catch (err) {
      console.error("Prediction error:", err);
      const errorMessage = "Error making prediction. Please try again.";
      setPrediction("Error");
      setPredictionConfidence(0);
      setTopPredictions([]);
      setGenAiPrediction(errorMessage);
      announcePrediction(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  const handleGenAiPredict = async () => {
    try {
      setGenAiLoading(true);
      setAiImage("");  // Clear previous image
      const canvas = canvasRef.current;
      
      // First, get the prediction if we don't have one
      if (!prediction) {
        await handlePredict();
        if (!prediction) {
          throw new Error('Please draw something and get a prediction first');
        }
      }

      // Show a loading message
      setGenAiPrediction('Analyzing your drawing with AI...');

      // Call the genai_guess endpoint via client
      const data = await PredictionClient.genAIGuess(
        canvas.toDataURL('image/png'),
        'What is this drawing? Choose one of: apple, airplane, cat, car, dog, flower, star, tree, umbrella, fish.'
      );
      
      if (data.guess) {
        // Update the prediction with the AI's guess
        setGenAiPrediction(`The AI thinks this is a: ${data.guess}`);
        
        // If the AI's guess is different from our initial prediction, update it
        if (data.guess.toLowerCase() !== prediction.toLowerCase()) {
          setPrediction(data.guess.toLowerCase());
          // You might want to update the confidence score as well
          setPredictionConfidence(0.9); // High confidence for the AI's guess
        }
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
      <MainContainer>
        {/* Header */}
        <Header>
          <HeaderLeft>
            <BrushIcon />
            <GlowingText>
              <ShinyText>Doodle Recognizer</ShinyText>
            </GlowingText>
          </HeaderLeft>
          <div>
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
              <Button fullWidth onClick={downloadAIDrawing} disabled={!aiImage}>⬇️ Download AI Image</Button>
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
              {/* Mobile quick actions placed right below the canvas */}
              <MobileActionsBar>
                <Button
                  onClick={handlePredict}
                  style={{ background: '#22c55e', color: '#fff', border: 'none' }}
                >
                  Analyze
                </Button>
                <Button
                  onClick={handleEnhance}
                  style={{ background: '#7c3aed', color: '#fff', border: 'none' }}
                  disabled={isEnhancing}
                >
                  {isEnhancing ? 'Generating…' : 'Generate Art'}
                </Button>
                <Button
                  onClick={clearCanvas}
                  style={{ background: '#ef4444', color: '#fff', border: 'none' }}
                >
                  Clear
                </Button>
              </MobileActionsBar>
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
                              <div style={{ fontSize: 16, fontWeight: 700, color: '#a5b4fc' }}>{prediction}</div>
                              <div style={{ opacity: 0.8 }}>Confidence: {Math.round((predictionConfidence || 0) * 100)}%</div>
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
            <div style={{ 
              background: 'rgba(99, 102, 241, 0.2)',
              borderRadius: '12px',
              padding: '16px',
              border: '1px solid rgba(99, 102, 241, 0.3)'
            }}>
              <h3 style={{ 
                color: '#ffffff',
                margin: '0 0 16px 0',
                fontWeight: '600',
                fontSize: '0.95rem',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                textAlign: 'center',
                paddingBottom: '8px',
                borderBottom: '1px solid rgba(99, 102, 241, 0.3)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                gap: '8px'
              }}>
                AI Analysis
              </h3>
              
              {genAiLoading ? (
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', padding: '16px 0' }}>
                  <LoadingSpinner />
                  <p style={{ color: '#a5b4fc', textAlign: 'center', margin: 0 }}>
                    Analyzing your drawing...
                  </p>
                </div>
              ) : (
                <div style={{ marginBottom: '16px' }}>
                  <p style={{ 
                    color: '#ffffff', 
                    padding: '16px',
                    background: 'rgba(6, 182, 212, 0.1)',
                    borderRadius: '8px',
                    borderLeft: '3px solid #06b6d4',
                    minHeight: '80px',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    textAlign: 'center',
                    margin: 0
                  }}>
                    {genAiPrediction || 'Draw something and click "Predict Doodle" to see AI analysis.'}
                  </p>
                </div>
              )}
                
              <Button
                fullWidth
                primary
                onClick={handleGenAiPredict}
                disabled={genAiLoading}
                style={{ marginBottom: '16px', textTransform: 'none' }}
              >
                {genAiLoading ? 'Analyzing...' : '🔍 Get AI Analysis'}
              </Button>
            </div>

            {/* Prediction Results Section */}
            <div style={{ 
              background: 'rgba(99, 102, 241, 0.2)',
              borderRadius: '12px',
              padding: '16px',
              border: '1px solid rgba(99, 102, 241, 0.3)',
              marginTop: '16px'
            }}>
              <h3 style={{ 
                color: '#e2e8f0',
                margin: '0 0 16px 0',
                fontWeight: '600',
                fontSize: '0.95rem',
                textTransform: 'uppercase',
                letterSpacing: '0.5px',
                textAlign: 'center',
                paddingBottom: '8px',
                borderBottom: '1px solid rgba(99, 102, 241, 0.3)'
              }}>
                Prediction Results
              </h3>
              
              <div style={{ marginBottom: '16px' }}>
                <p style={{ 
                  color: '#e2e8f0', 
                  padding: '16px',
                  background: 'rgba(99, 102, 241, 0.1)',
                  borderRadius: '8px',
                  borderLeft: '3px solid #6366f1',
                  minHeight: '60px',
                  display: 'flex',
                  flexDirection: 'column',
                  justifyContent: 'center',
                  textAlign: 'center',
                  margin: 0
                }}>
                  {prediction ? (
                    <>
                      <span style={{ 
                        fontSize: '1.1rem', 
                        fontWeight: '600',
                        marginBottom: '8px'
                      }}>
                        {prediction.charAt(0).toUpperCase() + prediction.slice(1)}
                      </span>
                      <span style={{ 
                        color: '#a5b4fc',
                        fontSize: '0.9rem'
                      }}>
                        Confidence: {Math.round(predictionConfidence * 100)}%
                      </span>
                    </>
                  ) : (
                    'Draw something and click "Predict Doodle" to see predictions.'
                  )}
                </p>
              </div>
            </div>


            {/* AI Generated Image Display */}
            {aiImage && (
              <div style={{ 
                padding: '16px', 
                borderRadius: '8px',
                background: 'linear-gradient(145deg, #1e1e2e 0%, #1a1a2e 100%)',
                boxShadow: '0 8px 32px rgba(0,0,0,0.3)',
                border: '1px solid rgba(99, 102, 241, 0.2)',
                backdropFilter: 'blur(10px)',
                marginTop: '12px'
              }}>
                <h3 style={{ color: 'white', marginBottom: '16px', fontWeight: 'bold', margin: '0 0 16px 0' }}>
                  🎭 AI Generated
                </h3>
                <img 
                  src={aiImage} 
                  alt="AI Generated Doodle" 
                  style={{ 
                    width: '100%', 
                    borderRadius: '8px',
                    border: '1px solid rgba(255,255,255,0.1)',
                    marginBottom: '12px'
                  }} 
                />
                <Button
                  fullWidth
                  onClick={downloadAIDrawing}
                  style={{ 
                    textTransform: 'none',
                    backgroundColor: colors.success,
                    color: 'white',
                    border: 'none'
                  }}
                >
                  💾 Download AI Image
                </Button>
              </div>
            )}
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
      <Modal open={showInfo} onClick={() => setShowInfo(false)}>
        <ModalContent onClick={(e) => e.stopPropagation()}>
          <CloseButton onClick={() => setShowInfo(false)} />
          
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '24px' }}>
            <span style={{ color: colors.primary, fontSize: '32px', marginRight: '16px' }}>ℹ️</span>
            <h2 style={{ 
              color: colors.primary, 
              fontFamily: 'Inter, sans-serif',
              fontWeight: 700,
              background: 'linear-gradient(135deg, #6366f1 0%, #a5b4fc 100%)',
              backgroundClip: 'text',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              margin: 0
            }}>
              About Doodle Recognizer
            </h2>
          </div>
          
          <p style={{ 
            lineHeight: 1.7, 
            marginBottom: '24px', 
            fontSize: '1.1rem',
            color: '#e2e8f0',
            margin: '0 0 24px 0'
          }}>
            This advanced doodle recognition app uses machine learning to identify your drawings and generate AI-powered artwork based on your sketches.
          </p>
          
          <div style={{ marginBottom: '24px' }}>
            <h3 style={{ 
              color: '#f472b6', 
              marginBottom: '16px', 
              fontWeight: 600,
              display: 'flex',
              alignItems: 'center',
              margin: '0 0 16px 0'
            }}>
              ✨ Features
            </h3>
            
            <div style={{ paddingLeft: 0, listStyle: 'none', margin: 0 }}>
              {[
                { icon: '🧠', text: 'Real-time drawing recognition using neural networks' },
                { icon: '🎨', text: 'AI-powered image generation based on your doodles' },
                { icon: '🖌', text: 'Customizable brush size and colors' },
                { icon: '💾', text: 'Download your drawings and AI-generated images' },
                { icon: '📱', text: 'Responsive canvas that adapts to your screen' }
              ].map((feature, index) => (
                <div 
                  key={index}
                  style={{ 
                    display: 'flex', 
                    alignItems: 'center', 
                    marginBottom: '12px',
                    padding: '16px',
                    borderRadius: '8px',
                    background: 'rgba(255,255,255,0.05)',
                    border: '1px solid rgba(255,255,255,0.1)',
                    transition: 'all 0.2s ease'
                  }}
                >
                  <span style={{ fontSize: '1.5rem', marginRight: '16px' }}>
                    {feature.icon}
                  </span>
                  <span style={{ color: '#cbd5e1', lineHeight: 1.6 }}>
                    {feature.text}
                  </span>
                </div>
              ))}
            </div>
          </div>
          
          <div style={{ 
            marginTop: '32px', 
            padding: '24px', 
            borderRadius: '12px',
            background: 'linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(244, 114, 182, 0.1) 100%)',
            border: '1px solid rgba(99, 102, 241, 0.2)',
            textAlign: 'center'
          }}>
            <p style={{ 
              color: '#a5b4fc', 
              fontSize: '1rem',
              fontWeight: 500,
              lineHeight: 1.6,
              margin: 0
            }}>
              Draw something on the canvas and click <strong>"Predict Doodle"</strong> to see the magic happen!
            </p>
          </div>
        </ModalContent>
      </Modal>
      
      <Footer>
        <div>© {new Date().getFullYear()} Doodle Recognizer. All rights reserved.</div>
        <div style={{ marginTop: '8px' }}>
          <a href="#" onClick={(e) => { e.preventDefault(); setShowInfo(true); }}>About</a>
          <a href="#" onClick={(e) => { e.preventDefault(); setActiveTab('privacy'); setShowTerms(true); }}>Privacy Policy</a>
          <a href="#" onClick={(e) => { e.preventDefault(); setActiveTab('terms'); setShowTerms(true); }}>Terms of Use</a>
          <a href="mailto:support@doodlerecognizer.com" onClick={(e) => e.stopPropagation()}>Contact</a>
        </div>
      </Footer>

      {/* Terms and Privacy Modal */}
      {showTerms && (
        <Modal onClick={() => setShowTerms(false)}>
          <ModalContent onClick={e => e.stopPropagation()} style={{ maxWidth: '800px', maxHeight: '80vh', overflowY: 'auto' }}>
            <CloseButton onClick={() => setShowTerms(false)} />
            
            <div style={{ display: 'flex', marginBottom: '20px', borderBottom: '1px solid rgba(255,255,255,0.1)' }}>
              <button 
                onClick={() => setActiveTab('privacy')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: activeTab === 'privacy' ? colors.primary : colors.textSecondary,
                  padding: '12px 24px',
                  fontSize: '16px',
                  cursor: 'pointer',
                  borderBottom: activeTab === 'privacy' ? `2px solid ${colors.primary}` : 'none',
                  transition: 'all 0.3s ease'
                }}
              >
                Privacy Policy
              </button>
              <button 
                onClick={() => setActiveTab('terms')}
                style={{
                  background: 'none',
                  border: 'none',
                  color: activeTab === 'terms' ? colors.primary : colors.textSecondary,
                  padding: '12px 24px',
                  fontSize: '16px',
                  cursor: 'pointer',
                  borderBottom: activeTab === 'terms' ? `2px solid ${colors.primary}` : 'none',
                  transition: 'all 0.3s ease'
                }}
              >
                Terms of Use
              </button>
            </div>

            {activeTab === 'privacy' ? (
              <div>
                <h2 style={{ color: colors.primary, marginBottom: '20px' }}>Privacy Policy</h2>
                <p style={{ color: colors.text, lineHeight: '1.6', marginBottom: '16px' }}>
                  <strong>Last Updated:</strong> August 23, 2024
                </p>
                
                <SectionTitle>1. Information We Collect</SectionTitle>
                <p style={{ color: colors.text, lineHeight: '1.6', marginBottom: '16px' }}>
                  We collect the following types of information when you use our service:
                </p>
                <ul style={{ color: colors.text, paddingLeft: '20px', marginBottom: '20px' }}>
                  <li>Drawings and sketches you create on our platform</li>
                  <li>Prediction results and AI-generated content</li>
                  <li>Basic usage data and analytics</li>
                </ul>

                <SectionTitle>2. How We Use Your Information</SectionTitle>
                <p style={{ color: colors.text, lineHeight: '1.6', marginBottom: '16px' }}>
                  We use your information to:
                </p>
                <ul style={{ color: colors.text, paddingLeft: '20px', marginBottom: '20px' }}>
                  <li>Provide and improve our doodle recognition service</li>
                  <li>Train and enhance our AI models (anonymously and in aggregate)</li>
                  <li>Analyze usage patterns to improve user experience</li>
                  <li>Prevent fraud and ensure service security</li>
                </ul>

                <SectionTitle>3. Data Security</SectionTitle>
                <p style={{ color: colors.text, lineHeight: '1.6', marginBottom: '16px' }}>
                  We implement appropriate security measures to protect your data, including encryption and secure server infrastructure. However, no method of transmission over the Internet is 100% secure.
                </p>
              </div>
            ) : (
              <div>
                <h2 style={{ color: colors.primary, marginBottom: '20px' }}>Terms of Use</h2>
                <p style={{ color: colors.text, lineHeight: '1.6', marginBottom: '16px' }}>
                  <strong>Effective Date:</strong> August 23, 2024
                </p>

                <SectionTitle>1. Acceptance of Terms</SectionTitle>
                <p style={{ color: colors.text, lineHeight: '1.6', marginBottom: '16px' }}>
                  By accessing or using the Doodle Recognizer service, you agree to be bound by these Terms of Use and our Privacy Policy.
                </p>

                <SectionTitle>2. Intellectual Property</SectionTitle>
                <p style={{ color: colors.text, lineHeight: '1.6', marginBottom: '16px' }}>
                  All content, features, and functionality of the service, including but not limited to text, graphics, logos, and software, are owned by Doodle Recognizer and are protected by copyright and other intellectual property laws.
                </p>

                <SectionTitle>3. User Content</SectionTitle>
                <p style={{ color: colors.text, lineHeight: '1.6', marginBottom: '16px' }}>
                  You retain ownership of any drawings or content you create using our service. By using the service, you grant us a non-exclusive, royalty-free, worldwide license to use, reproduce, modify, and display such content for the purpose of providing and improving our services.
                </p>

                <SectionTitle>4. Limitation of Liability</SectionTitle>
                <p style={{ color: colors.text, lineHeight: '1.6', marginBottom: '16px' }}>
                  Doodle Recognizer shall not be liable for any indirect, incidental, special, consequential, or punitive damages resulting from your use of or inability to use the service.
                </p>

                <SectionTitle>5. Changes to Terms</SectionTitle>
                <p style={{ color: colors.text, lineHeight: '1.6', marginBottom: '16px' }}>
                  We reserve the right to modify these terms at any time. We will provide notice of any changes by updating the "Last Updated" date at the top of these terms.
                </p>
              </div>
            )}
          </ModalContent>
        </Modal>
      )}
    </AppContainer>
  );
}

export default App; 