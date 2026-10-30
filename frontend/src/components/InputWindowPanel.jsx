/**
 * InputWindowPanel — shows the 7-day consecutive input window from the status API.
 * Indicates which dates form the complete aligned window for model inference.
 */
export default function InputWindowPanel({ status, loading }) {
  if (loading || !status) return null;

  const window = status?.consecutive_input_window;
  if (!window || !window.dates || window.dates.length === 0) return null;

  return (
    <div className="panel">
      <div className="panel__header">
        <span className="panel__title">Input Window</span>
        <span className="panel__badge" style={
          window.is_window_complete
            ? {}
            : { background: '#fdf8ef', color: 'var(--color-status-warn)' }
        }>
          {window.length_days}/{window.required_days} days
        </span>
      </div>

      <div className="input-window">
        {window.dates.map(d => (
          <span key={d} className="input-window__day input-window__day--active">
            {d.slice(5)}
          </span>
        ))}
      </div>

      {!window.is_window_complete && (
        <p style={{
          fontSize: 'var(--font-size-sm)',
          color: 'var(--color-text-secondary)',
          marginTop: 'var(--space-3)',
          lineHeight: 1.5,
        }}>
          The model requires {window.required_days} consecutive days of aligned data.
          Currently {window.length_days} day{window.length_days !== 1 ? 's' : ''} available.
        </p>
      )}
    </div>
  );
}
