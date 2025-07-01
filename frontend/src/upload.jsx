import React, { useState, useEffect, useRef } from 'react';
import './upload.css';


import uploadIcon from './assets/upload-icon.png';
import deleteIcon from './assets/delete-icon.png';
import copyIcon from './assets/copy-icon.png';
import downloadIcon from './assets/download-icon.png';
import logo from './assets/logo_.png'

import loaderAnimation from './assets/loader1.json';
import Lottie from 'lottie-react';

const Upload = () => {
  const [imagePreview, setImagePreview] = useState(null);
  const [pdfFile, setPdfFile] = useState(null);
  const [responseData, setResponseData] = useState('');
  const [loading, setLoading] = useState(false);
  const [warning, setWarning] = useState('');
  const [typedText, setTypedText] = useState('');
  const fileInputRef = useRef(null);

  const fullText =
    "TTransform handwritten notes into clean digital text with Glyphic – powered by advanced OCR intelligence designed for clarity and accuracy";

  useEffect(() => {
    let index = 0;
    let intervalId;
    let timeoutId;

    const startTyping = () => {
      setTypedText('');
      index = 0;
      intervalId = setInterval(() => {
        setTypedText((prev) => prev + fullText.charAt(index));
        index++;
        if (index >= fullText.length) {
          clearInterval(intervalId);
          timeoutId = setTimeout(startTyping, 5000);
        }
      }, 20);
    };

    startTyping();

    return () => {
      clearInterval(intervalId);
      clearTimeout(timeoutId);
    };
  }, []);

  const handleImageUpload = (event) => {
    const file = event.target.files[0];
    if (file && file.type === 'application/pdf') {
      setPdfFile(file);
      setImagePreview(URL.createObjectURL(file));
      setWarning('');
    } else {
      setWarning('Please upload a valid PDF file.');
    }
  };

  const handleSubmit = async () => {
    if (!pdfFile) {
      setWarning('Please upload a PDF file before clicking Digitize!');
      return;
    }

    setWarning('');
    setLoading(true);
    const formData = new FormData();
    formData.append('pdf', pdfFile);

    try {
      const response = await fetch('/api/extract-table', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Something went wrong');

      setResponseData(data.extracted_data);
    } catch (error) {
      console.error(error);
      setWarning('Failed to extract table.');
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = () => {
    setImagePreview(null);
    setPdfFile(null);
    setResponseData('');
    setWarning('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <>

    <div className="logo-container">
      <img src={logo} alt="Logo" className="logo" />
    </div>

      <div className="upload-container">
        <p className="glyphic-description">{typedText}</p>

        <label className="upload-box">
          <input
            ref={fileInputRef}
            type="file"
            accept="application/pdf"
            onChange={handleImageUpload}
          />
          <div className="upload-placeholder">
            <img src={uploadIcon} alt="Upload Icon" className="upload-icon" />
            <p>Choose a PDF file or drag it here</p>
          </div>
        </label>

        <div className="digitize-section">
          <button className="digitize-button" onClick={handleSubmit} disabled={loading}>
            {loading ? 'Extracting...' : 'Digitize'}
          </button>
          {loading && (
            <div className="loader-container">
              <Lottie animationData={loaderAnimation} loop={true} />
            </div>
          )}
        </div>

        {warning && (
          <p style={{ color: 'red', marginTop: '10px', fontSize: '14px' }}>{warning}</p>
        )}
      </div>

      {(imagePreview || responseData) && (
        <div className="preview-response-container">
          {imagePreview && (
            <div className="image-preview-wrapper">
              <button className="delete-button" onClick={handleDelete}>
                <img src={deleteIcon} alt="Delete" className="delete-icon" />
              </button>
              <iframe src={imagePreview} title="PDF Preview" className="image-preview" />
            </div>
          )}

          {responseData && (
            <div className="response-wrapper">
              <div className="response-actions-fixed">
                <button
                  className="action-button"
                  onClick={() => navigator.clipboard.writeText(responseData)}
                >
                  <img src={copyIcon} alt="Copy" className="action-icon" />
                </button>
                <button
                  className="action-button"
                  onClick={() => {
                    const blob = new Blob([responseData], { type: 'text/plain' });
                    const url = URL.createObjectURL(blob);
                    const link = document.createElement('a');
                    link.href = url;
                    link.download = 'extracted_table.txt';
                    link.click();
                    URL.revokeObjectURL(url);
                  }}
                >
                  <img src={downloadIcon} alt="Download" className="action-icon" />
                </button>
              </div>

              <div className="response-box">
              <pre className="response-text">{responseData}</pre>

              </div>
            </div>
          )}
        </div>
      )}
    </>
  );
};

export default Upload;
