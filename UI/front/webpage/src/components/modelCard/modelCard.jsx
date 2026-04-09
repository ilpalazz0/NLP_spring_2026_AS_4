import './modelCard.css';

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(2);
}

function ModelCard({ model, active, onClick }) {
  return (
    <button type="button" className={`model-card ${active ? 'is-active' : ''}`} onClick={onClick}>
      <div className="model-card-top">
        <span className={`embedding-pill embedding-pill-${model.embedding_type}`}>
          {model.embedding_type.toUpperCase()}
        </span>
        <h3>{model.run_id}</h3>
      </div>

      <div className="model-card-metrics">
        <div>
          <label>Val F1</label>
          <strong>{formatMetric(model.best_val_f1)}</strong>
        </div>
        <div>
          <label>Test F1</label>
          <strong>{formatMetric(model.test_f1)}</strong>
        </div>
        <div>
          <label>Test EM</label>
          <strong>{formatMetric(model.test_em)}</strong>
        </div>
      </div>
    </button>
  );
}

export default ModelCard;
