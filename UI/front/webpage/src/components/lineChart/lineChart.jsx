import './lineChart.css';

function buildPoints(values, width, height, padding, min, max) {
  if (!values.length) return '';
  const range = max - min || 1;
  return values
    .map((value, index) => {
      const x = padding + (index * (width - padding * 2)) / Math.max(values.length - 1, 1);
      const y = height - padding - ((value - min) / range) * (height - padding * 2);
      return `${x},${y}`;
    })
    .join(' ');
}

function LineChart({ title, labels, series }) {
  const width = 680;
  const height = 260;
  const padding = 28;
  const validSeries = series.filter((item) => item.values?.length);
  const flattened = validSeries.flatMap((item) => item.values);

  if (!validSeries.length || !flattened.length) {
    return (
      <div className="chart-card">
        <div className="chart-header">
          <h3>{title}</h3>
        </div>
        <div className="chart-empty">No data available yet.</div>
      </div>
    );
  }

  const min = Math.min(...flattened);
  const max = Math.max(...flattened);

  return (
    <div className="chart-card">
      <div className="chart-header">
        <h3>{title}</h3>
        <div className="chart-legend">
          {validSeries.map((item) => (
            <div className="chart-legend-item" key={item.label}>
              <span className="legend-swatch" style={{ background: item.color }} />
              <span>{item.label}</span>
            </div>
          ))}
        </div>
      </div>

      <svg className="chart-svg" viewBox={`0 0 ${width} ${height}`} role="img" aria-label={title}>
        {[0, 1, 2, 3].map((tick) => {
          const y = padding + (tick * (height - padding * 2)) / 3;
          return <line key={tick} x1={padding} y1={y} x2={width - padding} y2={y} className="chart-grid-line" />;
        })}
        {validSeries.map((item) => (
          <polyline
            key={item.label}
            fill="none"
            stroke={item.color}
            strokeWidth="3"
            strokeLinejoin="round"
            strokeLinecap="round"
            points={buildPoints(item.values, width, height, padding, min, max)}
          />
        ))}
      </svg>

      <div className="chart-axis-labels">
        {labels.map((label) => (
          <span key={label}>{label}</span>
        ))}
      </div>
    </div>
  );
}

export default LineChart;
