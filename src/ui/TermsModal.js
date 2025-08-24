import React from 'react';
import { Modal, ModalContent, CloseButton, SectionTitle } from './Styled';
import colors from '../constants/colors';

function TermsModal({ open, onClose, activeTab, setActiveTab }) {
  if (!open) return null;
  return (
    <Modal onClick={onClose}>
      <ModalContent onClick={(e) => e.stopPropagation()} style={{ maxWidth: '800px', maxHeight: '80vh', overflowY: 'auto' }}>
        <CloseButton onClick={onClose} />
        
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
  );
}

export default TermsModal;
