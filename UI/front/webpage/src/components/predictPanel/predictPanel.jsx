import { useEffect, useState } from 'react';
import './predictPanel.css';

const samplePrompts = [
  {
    id: 'capital',
    title: 'Capital city',
    question: 'What is the capital of Azerbaijan?',
    context:
      'Azerbaijan is a country in the South Caucasus. Its capital is Baku, which lies on the western coast of the Caspian Sea.',
  },
  {
    id: 'currency',
    title: 'Currency',
    question: 'What currency is used in Azerbaijan?',
    context:
      'The official currency of Azerbaijan is the Azerbaijani manat. It is commonly abbreviated as AZN in financial and exchange-rate contexts.',
  },
  {
    id: 'landmark',
    title: 'Historic landmark',
    question: 'In which city is the Maiden Tower located?',
    context:
      'The Maiden Tower is one of Azerbaijan’s most famous historic landmarks. It stands in the Old City of Baku and is part of a UNESCO World Heritage site.',
  },
];

function PredictPanel({ selectedSummary, prediction, onPredict, loading }) {
  const [activeSampleId, setActiveSampleId] = useState(samplePrompts[0].id);
  const [question, setQuestion] = useState(samplePrompts[0].question);
  const [context, setContext] = useState(samplePrompts[0].context);
  const [maxAnswerLength, setMaxAnswerLength] = useState(30);

  useEffect(() => {
    setActiveSampleId(samplePrompts[0].id);
    setQuestion(samplePrompts[0].question);
    setContext(samplePrompts[0].context);
    setMaxAnswerLength(30);
  }, [selectedSummary?.run_id]);

  function applySample(sample) {
    setActiveSampleId(sample.id);
    setQuestion(sample.question);
    setContext(sample.context);
  }

  function handleSubmit(event) {
    event.preventDefault();
    onPredict({ question, context, max_answer_len: maxAnswerLength });
  }

  return (
    <section className="predict-panel">
      <div className="predict-panel-top">
        <div>
          <h2>Try a prediction</h2>
          <p className="predict-description">Use a preset example or type your own question and context.</p>
        </div>
        <div className="selected-model-chip">
          <span>Model</span>
          <strong>{selectedSummary?.run_id || 'No model selected'}</strong>
        </div>
      </div>

      <div className="sample-prompts-card">
        <div className="sample-prompts-header">
          <h3>Sample questions</h3>
          <span>Click to fill the fields</span>
        </div>

        <div className="sample-prompts-grid">
          {samplePrompts.map((sample) => (
            <button
              key={sample.id}
              type="button"
              className={`sample-prompt-button ${activeSampleId === sample.id ? 'is-active' : ''}`}
              onClick={() => applySample(sample)}
            >
              <strong>{sample.title}</strong>
              <span>{sample.question}</span>
            </button>
          ))}
        </div>
      </div>

      <form className="predict-form" onSubmit={handleSubmit}>
        <div className="input-group">
          <label htmlFor="question-input">Question</label>
          <textarea
            id="question-input"
            value={question}
            onChange={(event) => setQuestion(event.target.value)}
            rows={2}
            placeholder="Enter a question"
          />
        </div>

        <div className="input-group">
          <label htmlFor="context-input">Context</label>
          <textarea
            id="context-input"
            value={context}
            onChange={(event) => setContext(event.target.value)}
            rows={7}
            placeholder="Paste the passage that contains the answer"
          />
        </div>

        <div className="predict-form-footer">
          <button type="submit" className="predict-button" disabled={loading || !selectedSummary}>
            {loading ? 'Running...' : 'Get answer'}
          </button>
        </div>
      </form>

      <div className="prediction-result-card">
        <h3>Result</h3>
        {prediction ? (
          <div className="prediction-content">
            <div className="prediction-answer-block">
              <label>Answer</label>
              <strong>{prediction.answer || 'No answer extracted.'}</strong>
            </div>
            <div className="prediction-meta-grid">
              <div>
                <span>Confidence</span>
                <strong>{prediction.confidence.toFixed(6)}</strong>
              </div>
              <div>
                <span>Start token</span>
                <strong>{prediction.start_index}</strong>
              </div>
              <div>
                <span>End token</span>
                <strong>{prediction.end_index}</strong>
              </div>
              <div>
                <span>Char span</span>
                <strong>
                  {prediction.start_char == null || prediction.end_char == null
                    ? '—'
                    : `${prediction.start_char} → ${prediction.end_char}`}
                </strong>
              </div>
            </div>
          </div>
        ) : (
          <p className="prediction-placeholder">
            Run a prediction to see the extracted answer and span.
          </p>
        )}
      </div>
    </section>
  );
}

export default PredictPanel;
