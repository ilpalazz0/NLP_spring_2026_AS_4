import LineChart from '../lineChart/lineChart';
import './metricsPanel.css';

function formatMetric(value) {
  if (value === null || value === undefined || Number.isNaN(value)) return '—';
  return Number(value).toFixed(2);
}

function MetricsPanel({ detail, loading, selectedSummary }) {
  if (loading) {
    return <section className="metrics-panel"><div className="panel-loading">Loading model details...</div></section>;
  }

  if (!detail || !selectedSummary) {
    return <section className="metrics-panel"><div className="panel-loading">Select a model to inspect its metrics.</div></section>;
  }

  const labels = detail.metrics_history.map((item) => `E${item.epoch}`);
  const trainLoss = detail.metrics_history.map((item) => item.train_loss);
  const valLoss = detail.metrics_history.map((item) => item.val_loss);
  const testLoss = detail.metrics_history.map((item) => item.test_loss).filter((value) => value !== null && value !== undefined);
  const valEm = detail.metrics_history.map((item) => item.val_em);
  const valF1 = detail.metrics_history.map((item) => item.val_f1);
  const testEm = detail.metrics_history.map((item) => item.test_em).filter((value) => value !== null && value !== undefined);
  const testF1 = detail.metrics_history.map((item) => item.test_f1).filter((value) => value !== null && value !== undefined);

  return (
    <section className="metrics-panel">
      <div className="metrics-top-row">
        <div>
          <h2>{selectedSummary.run_id}</h2>
          <p className="metrics-description">Core training and evaluation metrics for the selected run.</p>
        </div>
      </div>

      <div className="metric-card-grid">
        <div className="metric-highlight-card">
          <label>Embedding</label>
          <strong>{selectedSummary.embedding_type.toUpperCase()}</strong>
        </div>
        <div className="metric-highlight-card">
          <label>Best epoch</label>
          <strong>{selectedSummary.best_epoch ?? '—'}</strong>
        </div>
        <div className="metric-highlight-card">
          <label>Val F1</label>
          <strong>{formatMetric(selectedSummary.best_val_f1)}</strong>
        </div>
        <div className="metric-highlight-card">
          <label>Test F1</label>
          <strong>{formatMetric(selectedSummary.test_f1)}</strong>
        </div>
        <div className="metric-highlight-card">
          <label>Test EM</label>
          <strong>{formatMetric(selectedSummary.test_em)}</strong>
        </div>
      </div>

      <div className="metrics-two-col">
        <LineChart
          title="Loss"
          labels={labels}
          series={[
            { label: 'Train', values: trainLoss, color: '#4f8cff' },
            { label: 'Validation', values: valLoss, color: '#6dd3c7' },
            { label: 'Test', values: testLoss, color: '#f5a35d' },
          ]}
        />
        <LineChart
          title="EM / F1"
          labels={labels}
          series={[
            { label: 'Val EM', values: valEm, color: '#4f8cff' },
            { label: 'Val F1', values: valF1, color: '#6dd3c7' },
            { label: 'Test EM', values: testEm, color: '#f28a94' },
            { label: 'Test F1', values: testF1, color: '#8b7df5' },
          ]}
        />
      </div>

      <div className="metrics-table-card">
        <h3>Epoch history</h3>
        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Epoch</th>
                <th>Train</th>
                <th>Val</th>
                <th>Val EM</th>
                <th>Val F1</th>
                <th>Test EM</th>
                <th>Test F1</th>
              </tr>
            </thead>
            <tbody>
              {detail.metrics_history.map((item) => (
                <tr key={item.epoch}>
                  <td>{item.epoch}</td>
                  <td>{item.train_loss.toFixed(4)}</td>
                  <td>{item.val_loss.toFixed(4)}</td>
                  <td>{item.val_em.toFixed(2)}</td>
                  <td>{item.val_f1.toFixed(2)}</td>
                  <td>{item.test_em == null ? '—' : item.test_em.toFixed(2)}</td>
                  <td>{item.test_f1 == null ? '—' : item.test_f1.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

export default MetricsPanel;
