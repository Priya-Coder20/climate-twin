/**
 * API service for Maharashtra Climate Twin
 * Centralises all backend communication with proper error handling.
 */

const API_BASE = import.meta.env.VITE_API_BASE || 'http://127.0.0.1:8000';

/**
 * Generic fetch wrapper with timeout, JSON parsing, and error normalisation.
 */
async function request(path, opts = {}) {
  const url = `${API_BASE}${path}`;
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 15_000);

  try {
    const res = await fetch(url, { signal: controller.signal, ...opts });
    clearTimeout(timeout);

    if (!res.ok) {
      const body = await res.text().catch(() => '');
      throw new Error(`HTTP ${res.status}: ${body || res.statusText}`);
    }

    const contentType = res.headers.get('content-type') || '';
    if (contentType.includes('application/json')) {
      return await res.json();
    }
    // For binary responses (e.g. map PNGs), return the blob
    return await res.blob();
  } catch (err) {
    clearTimeout(timeout);
    if (err.name === 'AbortError') {
      throw new Error('Request timed out after 15 seconds');
    }
    throw err;
  }
}

/** GET /api/health */
export async function fetchHealth() {
  return request('/api/health');
}

/** GET /api/status */
export async function fetchStatus() {
  return request('/api/status');
}

/** GET /api/prediction */
export async function fetchPrediction(includeHistorical = false) {
  const qs = includeHistorical ? '?include_historical=true' : '';
  return request(`/api/prediction${qs}`);
}

/**
 * GET /api/maps/{mapType}
 * Returns an object-URL string pointing to the PNG blob.
 * mapType must be 'tmax' or 'tmin'.
 */
export async function fetchMapImage(mapType) {
  const blob = await request(`/api/maps/${mapType}`);
  return URL.createObjectURL(blob);
}

/**
 * Build the static URL for a map image (useful for <img> src).
 */
export function getMapUrl(mapType) {
  return `${API_BASE}/api/maps/${mapType}`;
}
