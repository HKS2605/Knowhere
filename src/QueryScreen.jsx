import React, { useState, useRef } from 'react';

// ACCENT COLOR — swapped from generic SaaS-blue (#4A6FA5) to terracotta,
// which reads as "earth observation / remote sensing" instead of "generic AI dashboard".
// If you want the olive/NDVI-green option instead, swap these two values for:
//   ACCENT = '#7A8B5C'   ACCENT_HOVER = '#647049'
const ACCENT = '#D97757';
const ACCENT_HOVER = '#C4623F';

export default function QueryScreen({ onAnalyze }) {
  const [primaryImage, setPrimaryImage] = useState(null);
  const [secondaryImage, setSecondaryImage] = useState(null);
  const [enableSecondImage, setEnableSecondImage] = useState(false);
  const [query, setQuery] = useState('');

  const primaryInputRef = useRef(null);
  const secondaryInputRef = useRef(null);

  const sampleQueries = [
    "Has the built-up area increased?",
    "Detect flooded regions using SAR coherence",
    "What land cover types are present?"
  ];

  const handleDrop = (e, setImage) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setImage(e.dataTransfer.files[0]);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const isFormValid = primaryImage !== null && query.trim().length > 0;

  const handleSubmit = () => {
    if (!isFormValid) return;
    if (onAnalyze) {
      onAnalyze(primaryImage, enableSecondImage ? secondaryImage : null, query);
    }
  };

  const renderDropZone = (image, setImage, inputRef, label) => (
    <div
      onDrop={(e) => handleDrop(e, setImage)}
      onDragOver={handleDragOver}
      onClick={() => inputRef.current.click()}
      className="border border-dashed border-[#262B35] bg-[#161920] hover:bg-[#1A1E26] rounded-md p-6 text-center cursor-pointer transition-colors"
    >
      <input
        type="file"
        ref={inputRef}
        onChange={(e) => e.target.files && setImage(e.target.files[0])}
        accept=".tif,.tiff,.png,.jpg,.jpeg"
        className="hidden"
      />
      {/* Centered Upload Icon */}
      <div className="flex justify-center mb-2">
        <svg
          className="w-6 h-6 text-[#8B93A1]"
          fill="none"
          stroke="currentColor"
          viewBox="0 0 24 24"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth="1.5"
            d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"
          />
        </svg>
      </div>

      <p className="text-sm text-[#8B93A1]">
        {image ? (
          <span className="text-[#E2E8F0] font-mono">{image.name}</span>
        ) : (
          <>
            Drop {label || "image"} or <span style={{ color: ACCENT }}>browse</span>
          </>
        )}
      </p>

      <p className="text-xs text-[#525B6C] mt-1 font-mono">
        GeoTIFF · PNG · JPEG
      </p>
    </div>
  );

  return (
    <div className="min-h-screen bg-[#0F1115] text-[#E2E8F0] p-8 font-sans">
      {/* HEADER */}
      <div className="flex items-center space-x-2 mb-10">
        <span
          style={{ color: ACCENT }}
          className="font-mono uppercase tracking-wider text-xs font-semibold"
        >
          SATQUERY
        </span>
        <span className="text-[#525B6C] font-mono text-xs">
          v0.4-beta
        </span>
      </div>

      {/* MAIN CONTENT */}
      <div className="max-w-[700px]">
        {/* Headings */}
        <h1 className="text-2xl font-bold tracking-tight text-white mb-2">
          Ask a question about your imagery
        </h1>
        <p className="text-sm text-[#8B93A1] mb-8 leading-relaxed">
          Upload optical or SAR imagery and query it in plain language. Supports VQA, change detection, and optical+SAR fusion.
        </p>

        {/* IMAGE SECTION */}
        <div className="mb-6 space-y-3">
          <label className="block text-[11px] font-mono uppercase tracking-wider text-[#8B93A1]">
            IMAGE
          </label>

          {/* Primary Drop Zone */}
          {renderDropZone(primaryImage, setPrimaryImage, primaryInputRef, "image")}

          {/* Toggle Second Image */}
          <div className="flex items-center space-x-3 pt-2">
            <button
              type="button"
              onClick={() => setEnableSecondImage(!enableSecondImage)}
              style={{ backgroundColor: enableSecondImage ? ACCENT : '#262B35' }}
              className="w-9 h-5 flex items-center rounded-full p-0.5 transition-colors duration-150"
            >
              <div
                className={`bg-white w-4 h-4 rounded-full transition-transform duration-150 transform ${
                  enableSecondImage ? 'translate-x-4' : 'translate-x-0'
                }`}
              />
            </button>
            <span className="text-xs text-[#8B93A1]">
              Add second image for change detection / optical+SAR fusion
            </span>
          </div>

          {/* Secondary Drop Zone (Conditional) */}
          {enableSecondImage && (
            <div className="pt-1">
              {renderDropZone(secondaryImage, setSecondaryImage, secondaryInputRef, "second image")}
            </div>
          )}
        </div>

        {/* DIVIDER */}
        <div className="border-t border-[#262B35] my-6" />

        {/* QUERY SECTION */}
        <div className="mb-6 space-y-3">
          <label className="block text-[11px] font-mono uppercase tracking-wider text-[#8B93A1]">
            QUERY
          </label>

          <textarea
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            rows={3}
            placeholder="e.g. Has the built-up area increased between these two dates?"
            className="w-full bg-[#161920] border border-[#262B35] rounded-md p-3 text-sm text-[#E2E8F0] placeholder-[#525B6C] focus:outline-none resize-none"
            style={{ borderColor: undefined }}
            onFocus={(e) => (e.target.style.borderColor = ACCENT)}
            onBlur={(e) => (e.target.style.borderColor = '#262B35')}
          />

          {/* Sample Query Chips */}
          <div className="flex flex-wrap gap-2 pt-1">
            {sampleQueries.map((sample, idx) => (
              <button
                key={idx}
                type="button"
                onClick={() => setQuery(sample)}
                className="text-xs text-[#8B93A1] border border-[#262B35] bg-[#161920] px-2.5 py-1 rounded transition-colors text-left"
                onMouseEnter={(e) => (e.target.style.borderColor = ACCENT)}
                onMouseLeave={(e) => (e.target.style.borderColor = '#262B35')}
              >
                {sample}
              </button>
            ))}
          </div>
        </div>

        {/* SUBMIT BUTTON */}
        <button
          type="button"
          disabled={!isFormValid}
          onClick={handleSubmit}
          style={
            isFormValid
              ? { backgroundColor: ACCENT }
              : undefined
          }
          onMouseEnter={(e) => {
            if (isFormValid) e.target.style.backgroundColor = ACCENT_HOVER;
          }}
          onMouseLeave={(e) => {
            if (isFormValid) e.target.style.backgroundColor = ACCENT;
          }}
          className={`w-full py-2.5 rounded text-sm font-medium transition-colors ${
            isFormValid
              ? 'text-white cursor-pointer'
              : 'bg-[#262B35] text-[#525B6C] cursor-not-allowed'
          }`}
        >
          Analyze
        </button>
      </div>
    </div>
  );
}
