import React, { useState } from 'react';

// Same accent as QueryScreen — keep these two files in sync.
const ACCENT = '#D97757';
const ACCENT_HOVER = '#C4623F';

export default function ResultScreen({
  results,
  isLoading = false,
  error = null,
  onRetry,
  onNewQuery,
  queryText = "what is this ?"
}) {
  const [isStepsOpen, setIsStepsOpen] = useState(false);

  const handleDownloadJSON = () => {
    if (!results) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(results, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", "satquery_results.json");
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const getConfidenceLevel = (score) => {
    if (score >= 80) return "HIGH";
    if (score >= 50) return "MED";
    return "LOW";
  };

  // 1. LOADING STATE
  if (isLoading) {
    return (
      <div className="min-h-screen bg-[#0F1115] text-[#E2E8F0] flex flex-col items-center justify-center p-6 font-sans">
        <div
          className="w-8 h-8 border-2 border-[#262B35] rounded-full animate-spin mb-4"
          style={{ borderTopColor: ACCENT }}
        />
        <span className="font-mono text-sm tracking-wider uppercase text-[#8B93A1]">
          Analyzing...
        </span>
      </div>
    );
  }

  // 2. ERROR STATE
  if (error) {
    return (
      <div className="min-h-screen bg-[#0F1115] text-[#E2E8F0] flex flex-col items-center justify-center p-6 font-sans">
        <div className="border border-[#262B35] bg-[#161920] p-3 rounded mb-4">
          <svg className="w-6 h-6 text-[#8B93A1]" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
          </svg>
        </div>
        <p className="text-sm text-[#8B93A1] mb-6">
          {typeof error === 'string' ? error : "Something went wrong. Please try again."}
        </p>
        <div className="flex space-x-3">
          {onRetry && (
            <button
              type="button"
              onClick={onRetry}
              style={{ backgroundColor: ACCENT }}
              onMouseEnter={(e) => (e.target.style.backgroundColor = ACCENT_HOVER)}
              onMouseLeave={(e) => (e.target.style.backgroundColor = ACCENT)}
              className="text-white px-4 py-2 rounded text-xs font-mono uppercase tracking-wider transition-colors"
            >
              Retry
            </button>
          )}
          {onNewQuery && (
            <button
              type="button"
              onClick={onNewQuery}
              className="border border-[#262B35] text-[#8B93A1] hover:text-[#E2E8F0] px-4 py-2 rounded text-xs font-mono uppercase tracking-wider transition-colors"
            >
              ← New query
            </button>
          )}
        </div>
      </div>
    );
  }

  const {
    inputImageUrl = '',
    overlayImageUrl = '',
    answer = '',
    confidence = 0,
    subScores = [],
    executionSteps = []
  } = results || {};

  // 3. MAIN RESULTS VIEW
  return (
    <div className="min-h-screen bg-[#0F1115] text-[#E2E8F0] p-8 font-sans">
      {/* TOP BAR */}
      <div className="flex items-center justify-between pb-6 mb-8 border-b border-[#262B35]">
        <div className="flex items-center space-x-2 font-mono text-xs">
          <span style={{ color: ACCENT }} className="uppercase tracking-wider font-semibold">SATQUERY</span>
          <span className="text-[#525B6C]">/</span>
          <span className="text-[#8B93A1] truncate max-w-md">{queryText}</span>
        </div>
        {onNewQuery && (
          <button
            type="button"
            onClick={onNewQuery}
            className="text-xs text-[#8B93A1] hover:text-[#E2E8F0] border border-[#262B35] bg-[#161920] px-3 py-1.5 rounded transition-colors font-mono"
            onMouseEnter={(e) => (e.target.style.borderColor = ACCENT)}
            onMouseLeave={(e) => (e.target.style.borderColor = '#262B35')}
          >
            ← New query
          </button>
        )}
      </div>

      {/* TWO-COLUMN LAYOUT */}
      <div className="flex flex-col lg:flex-row gap-8">

        {/* LEFT COLUMN (~65%) */}
        <div className="w-full lg:w-[65%] space-y-6">
          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-[#8B93A1] mb-3">
              IMAGE EVIDENCE
            </label>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div className="flex flex-col">
                <div className="w-full aspect-[4/3] bg-[#161920] border border-[#262B35] rounded-md overflow-hidden flex items-center justify-center">
                  {inputImageUrl ? (
                    <img src={inputImageUrl} alt="Input evidence" className="w-full h-full object-cover" />
                  ) : (
                    <span className="text-xs font-mono text-[#525B6C]">No input preview</span>
                  )}
                </div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-[#525B6C] mt-2 text-center">
                  INPUT
                </span>
              </div>

              <div className="flex flex-col">
                <div className="w-full aspect-[4/3] bg-[#161920] border border-[#262B35] rounded-md overflow-hidden flex items-center justify-center">
                  {overlayImageUrl ? (
                    <img src={overlayImageUrl} alt="Overlay analysis" className="w-full h-full object-cover" />
                  ) : (
                    <span className="text-xs font-mono text-[#525B6C]">No overlay preview</span>
                  )}
                </div>
                <span className="text-[11px] font-mono uppercase tracking-wider text-[#525B6C] mt-2 text-center">
                  OVERLAY
                </span>
              </div>
            </div>
          </div>

          <div>
            <label className="block text-[11px] font-mono uppercase tracking-wider text-[#8B93A1] mb-2">
              ANSWER
            </label>
            <div className="bg-[#161920] border border-[#262B35] rounded-md p-4">
              <p className="text-sm leading-relaxed text-[#E2E8F0]">
                {answer || "No response generated."}
              </p>
            </div>
          </div>
        </div>

        {/* RIGHT SIDEBAR (~35%) */}
        <div className="w-full lg:w-[35%]">
          <div className="bg-[#161920] border border-[#262B35] rounded-md p-5 space-y-6">

            {/* CONFIDENCE SECTION */}
            <div>
              <label className="block text-[11px] font-mono uppercase tracking-wider text-[#8B93A1] mb-2">
                CONFIDENCE
              </label>

              <div className="flex items-center space-x-2 mb-3">
                <span className="text-3xl font-bold font-mono text-white tracking-tight">
                  {confidence}%
                </span>
                <span className="text-[10px] font-mono uppercase bg-[#262B35] text-[#8B93A1] px-1.5 py-0.5 rounded border border-[#3A4150]">
                  {getConfidenceLevel(confidence)}
                </span>
              </div>

              <div className="w-full bg-[#262B35] h-1.5 rounded-full overflow-hidden mb-5">
                <div
                  className="h-full transition-all duration-300"
                  style={{ width: `${Math.min(Math.max(confidence, 0), 100)}%`, backgroundColor: ACCENT }}
                />
              </div>

              <div className="space-y-2">
                {subScores.map((scoreItem, idx) => (
                  <div key={idx} className="flex justify-between items-center text-xs">
                    <span className="text-[#8B93A1]">{scoreItem.label}</span>
                    <span className="font-mono font-bold" style={{ color: ACCENT }}>{scoreItem.value}</span>
                  </div>
                ))}
              </div>
            </div>

            <div className="border-t border-[#262B35]" />

            {/* AGENT EXECUTION SECTION */}
            <div>
              <button
                type="button"
                onClick={() => setIsStepsOpen(!isStepsOpen)}
                className="w-full flex items-center justify-between text-left focus:outline-none"
              >
                <span className="text-[11px] font-mono uppercase tracking-wider text-[#8B93A1]">
                  AGENT EXECUTION
                </span>
                <svg
                  className={`w-4 h-4 text-[#8B93A1] transition-transform duration-200 ${
                    isStepsOpen ? 'transform rotate-180' : ''
                  }`}
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M19 9l-7 7-7-7" />
                </svg>
              </button>

              {isStepsOpen && (
                <div className="mt-3 space-y-2 pt-2 border-t border-[#262B35]/50">
                  {executionSteps.length > 0 ? (
                    executionSteps.map((step, idx) => (
                      <div key={idx} className="flex items-center space-x-2 text-xs font-mono text-[#8B93A1]">
                        <span style={{ color: ACCENT }}>✓</span>
                        <span>{step.replace(/^✓\s*/, '')}</span>
                      </div>
                    ))
                  ) : (
                    <div className="text-xs font-mono text-[#525B6C]">No execution steps available.</div>
                  )}
                </div>
              )}
            </div>

            {/* DOWNLOAD BUTTON */}
            <button
              type="button"
              onClick={handleDownloadJSON}
              className="w-full flex items-center justify-center space-x-2 border border-[#262B35] bg-[#161920] hover:bg-[#1A1E26] text-[#8B93A1] hover:text-[#E2E8F0] py-2.5 rounded text-xs font-mono uppercase tracking-wider transition-colors"
              onMouseEnter={(e) => (e.currentTarget.style.borderColor = ACCENT)}
              onMouseLeave={(e) => (e.currentTarget.style.borderColor = '#262B35')}
            >
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="1.5" d="M4 16v1a2 2 0 002 2h12a2 2 0 002-2v-1M7 10l5 5m0 0l5-5m-5 5V3" />
              </svg>
              <span>Download results (JSON)</span>
            </button>

          </div>
        </div>

      </div>
    </div>
  );
}