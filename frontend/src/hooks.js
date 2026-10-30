import { useState, useEffect, useCallback } from 'react';

/**
 * Generic data-fetching hook with loading / error / data states.
 * @param {() => Promise<T>} fetchFn  – async function that returns data
 * @param {any[]} deps               – dependency array for re-fetching
 * @returns {{ data: T|null, loading: boolean, error: string|null, refetch: () => void }}
 */
export function useApiData(fetchFn, deps = []) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const result = await fetchFn();
      setData(result);
    } catch (err) {
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  useEffect(() => {
    load();
  }, [load]);

  return { data, loading, error, refetch: load };
}
