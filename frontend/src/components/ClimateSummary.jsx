/**
 * ClimateSummary — displays real climate values from the prediction endpoint.
 * Shows temperature statistics only when returned by the backend.
 * Clearly labels unavailable data rather than substituting examples.
 */
export default function ClimateSummary({ prediction, loading, error }) {
  if (loading) {
    return (
      <div className="panel">
        <div className="panel__header">
          <span className="panel__title">Climate Summary</span>
        </div>
        <div className="climate-grid">
          <div className="climate-card">
            <div className="skeleton skeleton--text" style={{ marginBottom: 8 }} />
            <div className="skeleton skeleton--value" />
          </div>
          <div className="climate-card">
            <div className="skeleton skeleton--text" style={{ marginBottom: 8 }} />
            <div className="skeleton skeleton--value" />
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="panel">
        <div className="panel__header">
          <span className="panel__title">Climate Summary</span>
        </div>
        <div className="error-banner">
          <span className="error-banner__icon">⚠</span>
          <span>Could not load climate summary: {error}</span>
        </div>
      </div>
    );
  }

  // Try to get the verified disk artifact data
  const artifact = prediction?.verified_disk_artifact;
  const hasArtifact = artifact?.available === true;
  const predDate = artifact?.prediction_date || null;
  const tmax = hasArtifact ? artifact.tmax : null;
  const tmin = hasArtifact ? artifact.tmin : null;

  return (
    <div className="panel">
      <div className="panel__header">
        <span className="panel__title">Climate Summary</span>
        {predDate && (
          <span className="panel__badge">{predDate}</span>
        )}
      </div>

      <div className="climate-grid">
        {/* Max Temperature */}
        <div className="climate-card">
          <div className="climate-card__label">Max Temp (mean)</div>
          {tmax ? (
            <>
              <div className="climate-card__value">
                {tmax.mean.toFixed(1)}
                <span className="climate-card__unit">°C</span>
              </div>
              <div className="climate-card__sub">
                Range: {tmax.min.toFixed(1)}–{tmax.max.toFixed(1)} °C
              </div>
            </>
          ) : (
            <div className="climate-card__value climate-card__value--unavailable">
              Not available
            </div>
          )}
        </div>

        {/* Min Temperature */}
        <div className="climate-card">
          <div className="climate-card__label">Min Temp (mean)</div>
          {tmin ? (
            <>
              <div className="climate-card__value">
                {tmin.mean.toFixed(1)}
                <span className="climate-card__unit">°C</span>
              </div>
              <div className="climate-card__sub">
                Range: {tmin.min.toFixed(1)}–{tmin.max.toFixed(1)} °C
              </div>
            </>
          ) : (
            <div className="climate-card__value climate-card__value--unavailable">
              Not available
            </div>
          )}
        </div>

        {/* Artifact Notice (only if showing historical artifact data) */}
        {hasArtifact && (
          <div className="climate-card climate-card--full" style={{ background: 'transparent', border: 'none', padding: 0 }}>
            <div className="forecast-artifact-notice">
              <span className="forecast-artifact-notice__icon">ℹ</span>
              <span>
                Values shown are from a previously evaluated model run on disk, not a live forecast.
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
