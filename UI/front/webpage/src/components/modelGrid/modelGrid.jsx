import ModelCard from '../modelCard/modelCard';
import './modelGrid.css';

function ModelGrid({ models, selectedRunId, onSelect, loading }) {
  return (
    <section className="model-grid-panel">
      <div className="panel-header-row">
        <div>
          <h2>Model runs</h2>
          <p>Select a run to view metrics and try predictions.</p>
        </div>
      </div>

      <div className="model-grid-list">
        {loading ? <div className="panel-empty-state">Loading runs...</div> : null}
        {!loading && models.length === 0 ? <div className="panel-empty-state">No trained models were found.</div> : null}
        {!loading && models.map((model) => (
          <ModelCard
            key={model.run_id}
            model={model}
            active={model.run_id === selectedRunId}
            onClick={() => onSelect(model.run_id)}
          />
        ))}
      </div>
    </section>
  );
}

export default ModelGrid;
