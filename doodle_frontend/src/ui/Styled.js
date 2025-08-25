/** UI components and animations extracted from App.js to keep JSX unchanged */
import { keyframes } from '@emotion/react';
import styled from '@emotion/styled';
import colors from '../constants/colors';

// Animations
export const glow = keyframes`
  from {
    text-shadow: 0 0 5px #fff, 0 0 10px #fff, 0 0 15px #0073e6, 0 0 20px #0073e6, 0 0 25px #0073e6, 0 0 30px #0073e6, 0 0 35px #0073e6;
  }
  to {
    text-shadow: 0 0 10px #fff, 0 0 20px #fff, 0 0 30px #00b4ff, 0 0 40px #00b4ff, 0 0 50px #00b4ff, 0 0 60px #00b4ff, 0 0 70px #00b4ff;
  }
  
  @media (max-width: 768px) {
    font-size: 0.6rem;
  }
`;

export const shiny = keyframes`
  0% { background-position: 0% center; }
  100% { background-position: 200% center; }
`;

export const pulse = keyframes`
  0%, 100% { opacity: 1; }
  50% { opacity: 0.7; }
`;

// Styled components
export const GlowingText = styled.h1`
  animation: ${glow} 1.5s ease-in-out infinite alternate;
  text-align: center;
  color: #fff;
  font-weight: bold;
  text-transform: uppercase;
  letter-spacing: 1px;
  margin: 0;
  font-size: 1.5rem;
`;

export const ShinyText = styled.span`
  background: linear-gradient(90deg, #ff8a00, #e52e71, #0073e6, #00b4ff, #ff8a00);
  background-size: 200% auto;
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  animation: ${shiny} 8s linear infinite;
`;

export const AppContainer = styled.div`
  height: 100vh;
  max-height: 100vh;
  overflow: hidden;
  background: ${colors.background};
  color: ${colors.text};
  display: flex;
  flex-direction: column;
  position: relative;
  
  @media (max-width: 768px) {
    height: auto;
    max-height: none;
    overflow: auto;
  }
`;

export const MainContainer = styled.div`
  flex: 1;
  padding: 4px;
  max-width: 1400px;
  margin: 0 auto;
  width: 100%;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  
  @media (max-width: 768px) {
    overflow: visible;
  }
`;

export const Header = styled.div`
  background: rgba(99, 102, 241, 0.2);
  backdrop-filter: blur(10px);
  border-radius: 16px;
  padding: 10px 12px;
  margin-bottom: 8px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  box-shadow: 0 4px 20px rgba(0,0,0,0.3);
  border: 1px solid rgba(99, 102, 241, 0.3);
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

export const HeaderLeft = styled.div`
  display: flex;
  align-items: center;
  gap: 16px;
`;

export const Footer = styled.footer`
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

export const BrushIcon = styled.div`
  color: ${colors.primary};
  font-size: 32px;
  filter: drop-shadow(0 0 8px rgba(99, 102, 241, 0.7));
  animation: ${pulse} 2s infinite;

  &::before {
    content: "🖌️";
  }
`;

export const InfoButton = styled.button`
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

export const ColorPicker = styled.div`
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 12px 0;
  padding: 8px;
  background: ${colors.surface2};
  border-radius: 8px;
`;

export const MainContent = styled.div`
  display: flex;
  flex: 1;
  gap: 6px;
  min-height: 0;
  max-height: calc(100vh - 120px);
  overflow: hidden;
  padding-bottom: 0;

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
    overflow: visible;
    padding-bottom: 0;
  }

  @media (max-width: 768px) {
    display: block;
    gap: 0;
    max-height: none;
    & > *:first-child,
    & > *:last-child { display: none; }
    & > *:nth-child(2) { flex: 1; }
  }

  &::-webkit-scrollbar { width: 4px; }
  &::-webkit-scrollbar-track { background: #f1f1f1; border-radius: 2px; }
  &::-webkit-scrollbar-thumb { background: #90caf9; border-radius: 2px; }
  &::-webkit-scrollbar-thumb:hover { background: #64b5f6; }
`;

export const Panel = styled.div`
  display: flex;
  flex-direction: column;
  gap: 6px;
  overflow-y: visible;
  height: calc(100vh - 150px);
  min-width: 250px;
  background: rgba(99, 102, 241, 0.15);
  backdrop-filter: blur(10px);
  border-radius: 20px;
  padding: 16px;
  border: 1px solid rgba(99, 102, 241, 0.2);
  box-shadow: 0 4px 20px rgba(0,0,0,0.15);
  transition: all 0.3s ease;
  
  &:hover {
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.25);
    border-color: rgba(99, 102, 241, 0.3);
  }
  
  @media (max-width: 768px) { display: none; }
`;

export const DarkPanel = styled.div`
  background: rgba(99, 102, 241, 0.2);
  backdrop-filter: blur(10px);
  box-shadow: 0 4px 20px rgba(0,0,0,0.2);
  border: 1px solid rgba(99, 102, 241, 0.25);
  border-radius: 16px;
  padding: 16px;
  margin-bottom: 16px;
  transition: all 0.3s ease;
  
  &:hover {
    background: rgba(99, 102, 241, 0.25);
    box-shadow: 0 6px 25px rgba(99, 102, 241, 0.3);
  }
`;

export const SectionTitle = styled.h3`
  color: #e2e8f0;
  margin-bottom: 12px;
  font-weight: 600;
  font-size: 0.95rem;
  letter-spacing: 0.5px;
  text-transform: uppercase;
  opacity: 0.9;
  margin: 0 0 12px 0;
`;

export const Button = styled.button`
  background: ${props => props.primary ? 'linear-gradient(45deg, #6366f1 30%, #4f46e5 90%)' : 'transparent'};
  color: ${props => props.primary ? '#fff' : colors.text};
  border: ${props => props.primary ? 'none' : `1px solid ${colors.primary}40`};
  border-radius: 8px;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 0.75rem;
  font-weight: 500;
  text-transform: none;
  transition: all 0.2s ease;
  width: ${props => props.fullWidth ? '100%' : 'auto'};
  margin-bottom: 8px;
  min-height: 32px;

  &:hover {
    background: ${props => props.primary ? 'linear-gradient(45deg, #4f46e5 30%, #4338ca 90%)' : 'rgba(99, 102, 241, 0.2)'};
    border-color: rgba(99, 102, 241, 0.7);
    transform: translateY(-1px);
  }

  &:active { transform: translateY(0); }
  &:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
`;

export const CanvasContainer = styled.div`
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
  
  @media (max-width: 768px) {
    border-radius: 0;
    border: none;
    padding: 0;
    background: ${colors.background};
  }
`;

export const CanvasWrapper = styled.div`
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  padding: 16px;
  box-sizing: border-box;
  position: relative;
  
  @media (max-width: 768px) {
    padding: 0;
    width: 100vw;
    height: 100vh; /* fallback */
    height: 100dvh; /* handle mobile browser UI correctly */
  }
`;

export const Canvas = styled.canvas`
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

export const ButtonGroup = styled.div`
  display: flex;
  gap: 16px;
  margin-top: 24px;
  width: 100%;
  max-width: ${props => props.maxWidth || '100%'};
  justify-content: center;
  padding: 0 20px;
  
  position: absolute;
  left: 50%;
  bottom: 8px;
  transform: translateX(-50%);
  z-index: 5;
  
  background: linear-gradient(180deg, rgba(30, 41, 59, 0.65), rgba(30, 41, 59, 0.5));
  backdrop-filter: blur(10px) saturate(140%);
  border-radius: 14px;
  padding: 10px 14px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.35), 0 0 0 1px rgba(148, 163, 184, 0.2) inset;

  &::before {
    content: "";
    position: absolute;
    inset: 0;
    padding: 1px; /* gradient border thickness */
    border-radius: inherit;
    background: linear-gradient(90deg, #ff8a00, #e52e71, #6366f1, #00b4ff, #ff8a00);
    -webkit-mask: linear-gradient(#000 0 0) content-box, linear-gradient(#000 0 0);
    -webkit-mask-composite: xor;
            mask-composite: exclude;
    pointer-events: none;
  }

  & > * {
    flex: 1;
    max-width: 260px;
    min-width: 180px;
    border-radius: 999px; /* pill */
    padding: 10px 18px;
    font-size: 0.85rem;
    box-shadow: 0 6px 18px rgba(99, 102, 241, 0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
  }

  & > *:hover:not(:disabled) {
    transform: translateY(-1px);
    box-shadow: 0 8px 24px rgba(99, 102, 241, 0.35);
  }
  
  @media (max-width: 768px) { display: none; }
`;

export const CanvasFrame = styled.div`
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
    width: calc(100vw - 24px);
    height: calc(100dvh - 24px);
    display: flex;
    align-items: center;
    justify-content: center;
  }
`;

export const HamburgerButton = styled.button`
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

export const MobileMenuOverlay = styled.div`
  position: fixed;
  inset: 0;
  background: rgba(0,0,0,0.8);
  backdrop-filter: blur(6px);
  z-index: 1100;
  display: flex;
  align-items: flex-end;
  @media (min-width: 769px) { display: none; }
`;

export const MobileMenuContent = styled.div`
  width: 100%;
  max-height: 85vh;
  background: ${colors.surface};
  border-top-left-radius: 16px;
  border-top-right-radius: 16px;
  border: 1px solid ${colors.surface2};
  padding: 16px;
  overflow-y: auto;
`;

export const MobileMenuHeader = styled.div`
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 12px;
  color: ${colors.text};
`;

export const MobilePredictionBox = styled.div`
  background: ${colors.surface2};
  border: 1px solid ${colors.surface2};
  border-radius: 12px;
  padding: 12px;
  margin-top: 12px;
  color: ${colors.text};
  font-size: 14px;
`;

export const MobileOnly = styled.div`
  display: none;
  @media (max-width: 768px) { display: block; }
`;

export const MobileActionsBar = styled.div`
  display: none;
  @media (max-width: 768px) {
    display: flex;
    position: static;
    padding: 10px 12px 0;
    background: transparent;
    gap: 10px;
    justify-content: center;
  }
`;

export const Modal = styled.div`
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

export const ModalContent = styled.div`
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

  &::-webkit-scrollbar { width: 8px; }
  &::-webkit-scrollbar-track { background: rgba(255,255,255,0.1); border-radius: 4px; }
  &::-webkit-scrollbar-thumb { background: rgba(99, 102, 241, 0.6); border-radius: 4px; }
  &::-webkit-scrollbar-thumb:hover { background: rgba(99, 102, 241, 0.8); }
`;

export const CloseButton = styled.button`
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

  &:hover { background: rgba(255,255,255,0.2); transform: scale(1.1); }
  &::before { content: "✕"; font-size: 16px; }
`;

export const LoadingSpinner = styled.div`
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

export const FeedbackBox = styled.div`
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
