import React from 'react';
import { Modal, ModalContent, CloseButton } from './Styled';
import colors from '../constants/colors';

function InfoModal({ open, onClose }) {
  return (
    <Modal open={open} onClick={onClose}>
      <ModalContent onClick={(e) => e.stopPropagation()}>
        <CloseButton onClick={onClose} />
        
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
        
        <div style={{ marginTop: '16px', marginBottom: '8px' }}>
          <h3 style={{ 
            color: '#60a5fa', 
            marginBottom: '12px', 
            fontWeight: 600,
            display: 'flex',
            alignItems: 'center'
          }}>
            🧩 Model Classes
          </h3>
          <p style={{ color: '#cbd5e1', margin: '0 0 12px 0', lineHeight: 1.6 }}>
            The model is trained to recognize these 10 classes:
          </p>
          <ul style={{ 
            display: 'grid', 
            gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))', 
            gap: '8px', 
            paddingLeft: 0, 
            listStyle: 'none', 
            margin: 0 
          }}>
            {['banana','apple','tree','car','smiley face','snake','ice cream','eye','star','envelope'].map((cls) => (
              <li 
                key={cls}
                style={{ 
                  background: 'rgba(255,255,255,0.06)', 
                  border: '1px solid rgba(255,255,255,0.1)', 
                  borderRadius: '8px', 
                  padding: '8px 12px', 
                  color: '#e2e8f0'
                }}
              >
                {cls}
              </li>
            ))}
          </ul>
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
  );
}

export default InfoModal;
