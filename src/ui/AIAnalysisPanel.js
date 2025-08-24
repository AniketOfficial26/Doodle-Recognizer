import React from 'react';
import { LoadingSpinner } from './Styled';

function AIAnalysisPanel({ genAiLoading, genAiPrediction }) {
  return (
    <div style={{ 
      background: 'rgba(99, 102, 241, 0.2)',
      borderRadius: '12px',
      padding: '16px',
      border: '1px solid rgba(99, 102, 241, 0.3)'
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
          <p style={{ color: '#f8fafc', textAlign: 'center', margin: 0, fontWeight: 800 }}>
            Analyzing your drawing...
          </p>
        </div>
      ) : (
        <div style={{ marginBottom: '16px' }}>
          <p style={{ 
            color: '#f8fafc', 
            padding: '16px',
            background: 'rgba(99, 102, 241, 0.18)',
            borderRadius: '8px',
            borderLeft: '3px solid #06b6d4',
            minHeight: '80px',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            textAlign: 'center',
            margin: 0,
            fontWeight: 800
          }}>
            {genAiPrediction || 'Draw something and click "Predict Doodle" to see AI analysis.'}
          </p>
        </div>
      )}
    </div>
  );
}

export default AIAnalysisPanel;
