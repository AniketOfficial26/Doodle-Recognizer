import React from 'react';
import { ProgressWrap, ProgressBar } from './Styled';

function PredictionResults({ prediction, predictionConfidence }) {
  return (
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
                fontWeight: 800,
                color: '#f8fafc',
                marginBottom: '8px'
              }}>
                {prediction.charAt(0).toUpperCase() + prediction.slice(1)}
              </span>
              <span style={{ 
                color: '#e2e8f0',
                fontSize: '0.95rem',
                fontWeight: 700,
                marginBottom: '8px'
              }}>
                Confidence: {Math.round((predictionConfidence || 0) * 100)}%
              </span>
              <ProgressWrap>
                <ProgressBar value={Math.round((predictionConfidence || 0) * 100)} />
              </ProgressWrap>
            </>
          ) : (
            'Draw something and click "Predict Doodle" to see predictions.'
          )}
        </p>
      </div>
    </div>
  );
}

export default PredictionResults;
