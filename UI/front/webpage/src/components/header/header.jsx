import './header.css';

function Header({ onRefresh, loading }) {
  return (
    <header className="showcase-header">
      <div className="showcase-copy">
        <h1>BiDAF + BERT model gallery</h1>
        <p className="showcase-subtitle">
          Compare saved runs and test them with your own question and context.
        </p>
      </div>

      <button type="button" className="refresh-button" onClick={onRefresh} disabled={loading}>
        {loading ? 'Refreshing...' : 'Refresh models'}
      </button>
    </header>
  );
}

export default Header;
