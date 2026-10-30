
import { useState, useEffect } from 'react';
import {
  MapContainer,
  TileLayer,
  ImageOverlay,
} from 'react-leaflet';
import 'leaflet/dist/leaflet.css';
import { getMapUrl } from '../api';

const MAHARASHTRA_BOUNDS = [
  [15.6, 72.6],
  [22.1, 80.9],
];

const MAP_CENTER = [19.0, 76.5];
const MAP_ZOOM = 7;

const MAP_LAYERS = [
  { key: 'tmax', label: 'Max Temperature' },
  { key: 'tmin', label: 'Min Temperature' },
];

export default function MapPanel() {
  const [activeLayer, setActiveLayer] = useState('tmax');
  const [overlayUrl, setOverlayUrl] = useState(null);
  const [overlayError, setOverlayError] = useState(null);
  const [loading, setLoading] = useState(true);
  const [opacity, setOpacity] = useState(0.7);
  const [retryKey, setRetryKey] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    let cancelled = false;
    let createdUrl = null;

    async function loadMap() {
      setLoading(true);
      setOverlayError(null);
      setOverlayUrl(null);

      try {
        const response = await fetch(getMapUrl(activeLayer), {
          signal: controller.signal,
        });

        if (!response.ok) {
          throw new Error(`Map request failed (${response.status})`);
        }

        const contentType = response.headers.get('content-type') || '';

        if (!contentType.includes('image/png')) {
          throw new Error('The server did not return a PNG map image.');
        }

        const blob = await response.blob();
        createdUrl = URL.createObjectURL(blob);

        if (!cancelled) {
          setOverlayUrl(createdUrl);
          setLoading(false);
        }
      } catch (error) {
        if (!cancelled && error.name !== 'AbortError') {
          setOverlayError(error.message || 'Unable to load the map.');
          setLoading(false);
        }
      }
    }

    loadMap();

    return () => {
      cancelled = true;
      controller.abort();

      if (createdUrl) {
        URL.revokeObjectURL(createdUrl);
      }
    };
  }, [activeLayer, retryKey]);

  return (
    <section
      className="map-section"
      aria-label="Maharashtra temperature map"
    >
      <div className="map-controls">
        <div style={{ marginBottom: '12px' }}>
          <div
            style={{
              fontSize: '1.05rem',
              fontWeight: 650,
              marginBottom: '4px',
            }}
          >
            Maharashtra Climate Map
          </div>

          <div
            style={{
              fontSize: '0.82rem',
              color: 'var(--color-text-muted)',
            }}
          >
            Spatial temperature output
          </div>
        </div>

        <div
          role="group"
          aria-label="Select temperature layer"
          style={{
            display: 'flex',
            gap: '8px',
            flexWrap: 'wrap',
          }}
        >
          {MAP_LAYERS.map((layer) => (
            <button
              key={layer.key}
              type="button"
              className={`map-controls__btn ${
                activeLayer === layer.key
                  ? 'map-controls__btn--active'
                  : ''
              }`}
              aria-pressed={activeLayer === layer.key}
              onClick={() => setActiveLayer(layer.key)}
            >
              {layer.label}
            </button>
          ))}
        </div>

        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '12px',
            marginTop: '12px',
            maxWidth: '360px',
          }}
        >
          <label
            htmlFor="temperature-overlay-opacity"
            style={{
              fontSize: '0.82rem',
              whiteSpace: 'nowrap',
            }}
          >
            Overlay opacity
          </label>

          <input
            id="temperature-overlay-opacity"
            type="range"
            min="0.2"
            max="1"
            step="0.1"
            value={opacity}
            onChange={(event) =>
              setOpacity(Number(event.target.value))
            }
            aria-valuetext={`${Math.round(opacity * 100)} percent`}
            style={{ flex: 1, minWidth: 80 }}
          />

          <span
            style={{
              fontSize: '0.82rem',
              minWidth: '38px',
              textAlign: 'right',
            }}
          >
            {Math.round(opacity * 100)}%
          </span>
        </div>
      </div>

      <MapContainer
        center={MAP_CENTER}
        zoom={MAP_ZOOM}
        className="map-container"
        zoomControl
        scrollWheelZoom
        style={{ height: '100%', width: '100%' }}
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {overlayUrl && !loading && (
          <ImageOverlay
            key={overlayUrl}
            url={overlayUrl}
            bounds={MAHARASHTRA_BOUNDS}
            opacity={opacity}
            zIndex={400}
          />
        )}
      </MapContainer>

      {loading && (
        <div
          className="map-placeholder"
          role="status"
          style={{
            position: 'absolute',
            inset: 0,
            zIndex: 450,
            background: 'rgba(247,245,242,0.72)',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '10px',
          }}
        >
          <div
            className="skeleton skeleton--block"
            style={{ width: 180, height: 20 }}
          />
          <span>Loading temperature map...</span>
        </div>
      )}

      {overlayError && !loading && (
        <div
          role="alert"
          style={{
            position: 'absolute',
            bottom: '16px',
            left: '16px',
            right: '16px',
            zIndex: 450,
            padding: '14px',
            borderRadius: '10px',
            background: 'rgba(255,248,245,0.97)',
            border: '1px solid #e6b8ad',
            color: '#8c3326',
          }}
        >
          <div style={{ fontWeight: 650, marginBottom: '5px' }}>
            Temperature map unavailable
          </div>

          <div style={{ fontSize: '0.85rem', marginBottom: '10px' }}>
            {overlayError}
          </div>

          <button
            type="button"
            onClick={() => setRetryKey((value) => value + 1)}
          >
            Try again
          </button>
        </div>
      )}
    </section>
  );
}
