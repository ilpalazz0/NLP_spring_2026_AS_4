import { useEffect, useMemo, useState } from 'react';
import { fetchModelDetail, fetchModels, predictAnswer, predictSentiment } from '../api/client';
import Header from '../components/header/header';
import ModelGrid from '../components/modelGrid/modelGrid';
import MetricsPanel from '../components/metricsPanel/metricsPanel';
import PredictPanel from '../components/predictPanel/predictPanel';
import './home.css';

const SENTIMENT_EXAMPLES = [
  "I absolutely loved this movie. The acting was great and the story was very engaging.",
  "The product was terrible and stopped working after two days.",
  "The hotel was clean, comfortable, and the staff were very friendly.",
  "I am disappointed with the service. It was slow and unprofessional.",
  "This book was amazing. I could not stop reading it.",
  "The food was cold and tasted awful."
];

function formatPercent(value) {
  return `${(Number(value || 0) * 100).toFixed(1)}%`;
}

function Home() {
  const [models, setModels] = useState([]);
  const [selectedRunId, setSelectedRunId] = useState('');
  const [selectedDetail, setSelectedDetail] = useState(null);
  const [loadingModels, setLoadingModels] = useState(true);
  const [loadingDetail, setLoadingDetail] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [error, setError] = useState('');

  const [sentimentText, setSentimentText] = useState('');
  const [sentimentLoading, setSentimentLoading] = useState(false);
  const [sentimentResult, setSentimentResult] = useState(null);

  useEffect(() => {
    async function bootstrap() {
      try {
        setLoadingModels(true);
        const modelData = await fetchModels();
        setModels(modelData);
        if (modelData.length > 0) {
          setSelectedRunId(modelData[0].run_id);
        }
      } catch (err) {
        setError(err.message || 'Failed to load data.');
      } finally {
        setLoadingModels(false);
      }
    }

    bootstrap();
  }, []);

  useEffect(() => {
    if (!selectedRunId) return;

    async function loadDetail() {
      try {
        setLoadingDetail(true);
        setPrediction(null);
        const detail = await fetchModelDetail(selectedRunId);
        setSelectedDetail(detail);
      } catch (err) {
        setError(err.message || 'Failed to load model detail.');
      } finally {
        setLoadingDetail(false);
      }
    }

    loadDetail();
  }, [selectedRunId]);

  const selectedSummary = useMemo(
    () => models.find((item) => item.run_id === selectedRunId) || null,
    [models, selectedRunId]
  );

  async function handleRefresh() {
    try {
      setError('');
      setLoadingModels(true);
      const modelData = await fetchModels(true);
      setModels(modelData);
      if (modelData.length && !modelData.some((item) => item.run_id === selectedRunId)) {
        setSelectedRunId(modelData[0].run_id);
      }
    } catch (err) {
      setError(err.message || 'Failed to refresh models.');
    } finally {
      setLoadingModels(false);
    }
  }

  async function handlePredict(formValues) {
    try {
      setError('');
      setPredicting(true);
      const result = await predictAnswer({
        run_id: selectedRunId,
        question: formValues.question,
        context: formValues.context,
        max_answer_len: formValues.max_answer_len,
      });
      setPrediction(result);
      const refreshed = await fetchModelDetail(selectedRunId);
      setSelectedDetail(refreshed);
    } catch (err) {
      setError(err.message || 'Prediction failed.');
    } finally {
      setPredicting(false);
    }
  }

  async function handleSentimentPredict() {
    try {
      setError('');
      setSentimentLoading(true);
      const result = await predictSentiment({ text: sentimentText });
      setSentimentResult(result);
    } catch (err) {
      setError(err.message || 'Sentiment prediction failed.');
    } finally {
      setSentimentLoading(false);
    }
  }

  return (
    <div className="home-page">
      <Header onRefresh={handleRefresh} loading={loadingModels} />

      {error ? <div className="home-alert">{error}</div> : null}

      <div className="home-layout">
        <ModelGrid
          models={models}
          loading={loadingModels}
          selectedRunId={selectedRunId}
          onSelect={setSelectedRunId}
        />

        <div className="home-main-column">
          <MetricsPanel
            detail={selectedDetail}
            loading={loadingDetail}
            selectedSummary={selectedSummary}
          />
          <PredictPanel
            selectedSummary={selectedSummary}
            prediction={prediction}
            onPredict={handlePredict}
            loading={predicting}
          />
        </div>
      </div>

      <section className="sentiment-panel">
        <div className="sentiment-panel__header">
          <div>
            <p className="sentiment-panel__eyebrow">Sentiment analysis</p>
            <h2>DistilBERT sentiment model</h2>
            <p className="sentiment-panel__description">
              Analyze Azerbaijani review text with your fine-tuned sequence-classification model.
            </p>
          </div>
        </div>

        <div className="sentiment-panel__samples">
          {SENTIMENT_EXAMPLES.map((sample) => (
            <button
              key={sample}
              type="button"
              className="sentiment-sample"
              onClick={() => setSentimentText(sample)}
            >
              {sample}
            </button>
          ))}
        </div>

        <textarea
          className="sentiment-textarea"
          rows={5}
          placeholder="Mətni bura yazın..."
          value={sentimentText}
          onChange={(event) => setSentimentText(event.target.value)}
        />

        <div className="sentiment-panel__actions">
          <button
            type="button"
            className="sentiment-submit"
            onClick={handleSentimentPredict}
            disabled={sentimentLoading || !sentimentText.trim()}
          >
            {sentimentLoading ? 'Analyzing...' : 'Analyze sentiment'}
          </button>
        </div>

        {sentimentResult ? (
          <div className="sentiment-result">
            <div className="sentiment-result__top">
              <span className={`sentiment-badge sentiment-badge--${sentimentResult.label}`}>
                {sentimentResult.label}
              </span>
              <strong>{formatPercent(sentimentResult.confidence)}</strong>
            </div>

            <div className="sentiment-score-list">
              {sentimentResult.scores.map((item) => (
                <div key={item.label} className="sentiment-score-row">
                  <span>{item.label}</span>
                  <span>{formatPercent(item.score)}</span>
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </section>
    </div>
  );
}

export default Home;
