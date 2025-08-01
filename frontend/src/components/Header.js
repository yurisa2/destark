import React from 'react';

function Header({ title }) {
  return (
    <header className="header">
      <div className="container">
        <h1>{title}</h1>
        <p>Fuzzy Inference System for Environmental Assessment</p>
      </div>
    </header>
  );
}

export default Header; 