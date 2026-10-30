/**
 * Header – compact top navbar with title, system status indicator, and latest data date.
 */
export default function Header({ health, status, healthLoading }) {
  // Derive status label
  let statusClass = 'status-dot--loading';
  let statusLabel = 'Loading';

  if (!healthLoading && health) {
    if (health.status === 'healthy') {
      // Check pipeline status for a more nuanced indicator
      const pipelineStatus = status?.status;
      if (pipelineStatus === 'aligned') {
        statusClass = 'status-dot--healthy';
        statusLabel = 'Aligned';
      } else if (pipelineStatus === 'stale_inputs' || pipelineStatus === 'insufficient') {
        statusClass = 'status-dot--stale';
        statusLabel = 'Stale';
      } else {
        statusClass = 'status-dot--healthy';
        statusLabel = 'Online';
      }
    } else {
      statusClass = 'status-dot--error';
      statusLabel = 'Error';
    }
  } else if (!healthLoading && !health) {
    statusClass = 'status-dot--error';
    statusLabel = 'Offline';
  }

  // Latest common date from status API
  const latestDate = status?.latest_common_date?.processed_channels_all_10;

  return (
    <header className="header">
      <div className="header__brand">
        <svg className="header__logo" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
          <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 17.93c-3.95-.49-7-3.85-7-7.93 0-.62.08-1.21.21-1.79L9 15v1c0 1.1.9 2 2 2v1.93zm6.9-2.54c-.26-.81-1-1.39-1.9-1.39h-1v-3c0-.55-.45-1-1-1H8v-2h2c.55 0 1-.45 1-1V7h2c1.1 0 2-.9 2-2v-.41c2.93 1.19 5 4.06 5 7.41 0 2.08-.8 3.97-2.1 5.39z" fill="currentColor" opacity="0.85"/>
        </svg>
        <h1 className="header__title">Maharashtra Climate Twin</h1>
      </div>

      <div className="header__meta">
        {latestDate && (
          <span className="header__date">
            Data through {latestDate}
          </span>
        )}
        <span className={`status-dot ${statusClass}`}>
          {statusLabel}
        </span>
      </div>
    </header>
  );
}
