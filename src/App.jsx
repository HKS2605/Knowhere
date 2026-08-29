import React, { useState } from 'react';
import QueryScreen from './QueryScreen';
import ResultScreen from './ResultScreen';

export default function App() {
  const [view, setView] = useState('query'); // 'query' | 'results'
  const [queryText, setQueryText] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);

  const handleAnalyze = (primaryImage, secondaryImage, query) => {
    setQueryText(query);
    setView('results');
    setIsLoading(true);
    setError(null);

    // TEMPORARY: fake response so you can see ResultScreen render right now,
    // before M1's real API is wired in. Replace this whole block with the
    // real fetch() call once M1's endpoint is live.
    setTimeout(() => {
      setIsLoading(false);
      setResults({
        inputImageUrl: URL.createObjectURL(primaryImage),
        overlayImageUrl: URL.createObjectURL(primaryImage),
        answer: "Built-up area has increased by approximately 12.4% between the two acquisition dates.",
        confidence: 84,
        subScores: [
          { label: "Change detection", value: "0.91" },
          { label: "Area estimation", value: "0.78" },
          { label: "Temporal alignment", value: "0.83" }
        ],
        executionSteps: [
          "Input validated",
          "Query → Change Analysis",
          "Change Model executed",
          "Evidence generated"
        ]
      });
    }, 1200);
  };

  const handleNewQuery = () => {
    setView('query');
    setResults(null);
    setError(null);
  };

  const handleRetry = () => {
    setError(null);
    setIsLoading(true);
    setTimeout(() => setIsLoading(false), 1000);
  };

  if (view === 'results') {
    return (
      <ResultScreen
        results={results}
        isLoading={isLoading}
        error={error}
        onRetry={handleRetry}
        onNewQuery={handleNewQuery}
        queryText={queryText}
      />
    );
  }

  return <QueryScreen onAnalyze={handleAnalyze} />;
}