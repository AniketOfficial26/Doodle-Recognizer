import React from 'react';
import { Footer } from './Styled';

function FooterBar({ onAbout, onPrivacy, onTerms }) {
  return (
    <Footer>
      <div>© {new Date().getFullYear()} Doodle Recognizer. All rights reserved.</div>
      <div style={{ marginTop: '8px' }}>
        <a href="#" onClick={(e) => { e.preventDefault(); onAbout(); }}>About</a>
        <a href="#" onClick={(e) => { e.preventDefault(); onPrivacy(); }}>Privacy Policy</a>
        <a href="#" onClick={(e) => { e.preventDefault(); onTerms(); }}>Terms of Use</a>
        <a href="mailto:support@doodlerecognizer.com" onClick={(e) => e.stopPropagation()}>Contact</a>
      </div>
    </Footer>
  );
}

export default FooterBar;
