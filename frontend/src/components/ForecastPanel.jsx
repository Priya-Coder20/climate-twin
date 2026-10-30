/**
 * ForecastPanel — displays the prediction state and verified artifact data.
 *
 * When live forecast is unavailable (which is the current state because
 * MOSDAC satellite observations have stopped), shows a professional message
 * explaining why. Also renders the historical evaluation artifact when present.
 */
export default function ForecastPanel({ prediction, loading, error }) {
  if (loading) {
    return (
      <div className="panel">
        <div className="panel__header">
          <span className="panel__title">Forecast</span>
        </div>
        <div className="forecast-block">
          <div className="skeleton skeleton--block" />
          <div className="skeleton skeleton--block" />
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="panel">
        <div className="panel__header">
          <span className="panel__title">Forecast</span>
        </div>
        <div className="error-banner">
          <span className="error-banner__icon">⚠</span>
          <span>Could not retrieve forecast data: {error}</span>
        </div>
      </div>
    );
  }

  const isAvailable = prediction?.available === true;
  const artifact = prediction?.verified_disk_artifact;
  const hasArtifact = artifact?.available === true;
  const alignment = prediction?.alignment_summary;

  return (
    <div className="panel">
      <div className="panel__header">
        <span className="panel__title">Forecast</span>
        <span className="panel__badge" style={
          isAvailable
            ? {}
            : { background: '#fdf8ef', color: 'var(--color-status-warn)' }
        }>
          {isAvailable ? 'Live' : 'Unavailable'}
        </span>
      </div>

      <div className="forecast-block">
        {/* Live forecast status */}
        {!isAvailable && (
          <div className="forecast-unavailable">
            <div className="forecast-unavailable__title">
              Live forecast is currently unavailable
            </div>
            <p className="forecast-unavailable__reason">
              {alignment?.reason || prediction?.message || 'Multi-source input data is not fully synchronized.'}
            </p>
          </div>
        )}

        {/* Verified historical artifact */}
        {hasArtifact && (
          <>
            <div className="forecast-artifact-notice">
              <span className="forecast-artifact-notice__icon">📋</span>
              <span>
                Historical evaluation artifact for <strong>{artifact.prediction_date}</strong> —
                {' '}generated from a {artifact.input_dates?.length || 7}-day input window
                using <em>{artifact.model}</em>.
              </span>
            </div>

            <div className="forecast-stat-grid">
              <div className="forecast-stat">
                <div className="forecast-stat__label">T-Max (mean)</div>
                <div className="forecast-stat__value" style={{ color: 'var(--color-status-err)' }}>
                  {artifact.tmax?.mean?.toFixed(1) ?? '—'}°C
                </div>
                {artifact.tmax && (
                  <div className="forecast-stat__range">
                    {artifact.tmax.min.toFixed(1)}–{artifact.tmax.max.toFixed(1)} °C
                  </div>
                )}
              </div>

              <div className="forecast-stat">
                <div className="forecast-stat__label">T-Min (mean)</div>
                <div className="forecast-stat__value" style={{ color: 'var(--color-status-info)' }}>
                  {artifact.tmin?.mean?.toFixed(1) ?? '—'}°C
                </div>
                {artifact.tmin && (
                  <div className="forecast-stat__range">
                    {artifact.tmin.min.toFixed(1)}–{artifact.tmin.max.toFixed(1)} °C
                  </div>
                )}
              </div>
            </div>

            {/* Input window dates */}
            {artifact.input_dates && artifact.input_dates.length > 0 && (
              <div>
                <div style={{ fontSize: 'var(--font-size-xs)', color: 'var(--color-text-muted)', marginBottom: 'var(--space-2)', fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.04em' }}>
                  Model Input Window
                </div>
                <div className="input-window">
                  {artifact.input_dates.map(d => (
                    <span key={d} className="input-window__day input-window__day--active">
                      {d.slice(5)}
                    </span>
                  ))}
                  <span className="input-window__day" style={{ background: 'var(--color-teal-100)', borderColor: 'var(--color-teal-400)', color: 'var(--color-teal-600)', fontWeight: 600 }}>
                    → {artifact.prediction_date?.slice(5)}
                  </span>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}
