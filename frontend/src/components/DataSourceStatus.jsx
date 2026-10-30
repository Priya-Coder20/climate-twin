/**
 * DataSourceStatus — shows the latest available date for each data source.
 * Derived entirely from the /api/status response.
 */
export default function DataSourceStatus({ status, loading, error }) {
  if (loading) {
    return (
      <div className="panel">
        <div className="panel__header">
          <span className="panel__title">Data Sources</span>
        </div>
        <div className="source-list">
          {[1, 2, 3].map(i => (
            <div className="source-row" key={i}>
              <div className="skeleton skeleton--text" style={{ width: 80 }} />
              <div className="skeleton skeleton--text" style={{ width: 90 }} />
            </div>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="panel">
        <div className="panel__header">
          <span className="panel__title">Data Sources</span>
        </div>
        <div className="error-banner">
          <span className="error-banner__icon">⚠</span>
          <span>Could not load source status.</span>
        </div>
      </div>
    );
  }

  const sources = status?.latest_available_dates || {};
  const sourceEntries = [
    { key: 'mosdac', label: 'MOSDAC (Satellite)' },
    { key: 'era5',   label: 'ERA5 (Reanalysis)' },
    { key: 'imd',    label: 'IMD (Ground Obs.)' },
  ];

  return (
    <div className="panel">
      <div className="panel__header">
        <span className="panel__title">Data Sources</span>
      </div>
      <div className="source-list">
        {sourceEntries.map(src => (
          <div className="source-row" key={src.key}>
            <span className="source-row__name">{src.label}</span>
            <span className="source-row__date">
              {sources[src.key] || '—'}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}
