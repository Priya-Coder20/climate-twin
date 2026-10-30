
import { useEffect, useMemo, useState } from "react";
import {
  MapContainer,
  TileLayer,
  GeoJSON,
  useMap,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";

const BOUNDARY_URL = "/data/maharashtra-districts.geojson";
const API_BASE_URL = "http://127.0.0.1:8000/api";

function normalizeVariable(variable) {
  const value = String(variable ?? "max_temperature")
    .toLowerCase()
    .trim();

  if (
    [
      "min_temperature",
      "minimum temperature",
      "minimum",
      "min",
      "tmin",
    ].includes(value)
  ) {
    return "tmin";
  }

  if (
    ["rainfall", "rain", "precipitation", "precip"].includes(value)
  ) {
    return "rainfall";
  }

  return "tmax";
}

function FitMaharashtra({ data }) {
  const map = useMap();

  useEffect(() => {
    if (!data) return;

    const bounds = L.geoJSON(data).getBounds();

    if (bounds.isValid()) {
      map.fitBounds(bounds, {
        padding: [18, 18],
        maxZoom: 7,
      });
    }
  }, [data, map]);

  return null;
}

function getColor(value, min, max, isRainfall = false) {
  if (!Number.isFinite(value)) return "#d9ded9";

  const stops = isRainfall
    ? [
        { p: 0, color: [255, 247, 188] },
        { p: 0.25, color: [254, 217, 118] },
        { p: 0.5, color: [254, 153, 41] },
        { p: 0.75, color: [236, 112, 20] },
        { p: 1, color: [153, 52, 4] },
      ]
    : [
        { p: 0, color: [49, 130, 189] },
        { p: 0.25, color: [107, 174, 214] },
        { p: 0.5, color: [254, 224, 139] },
        { p: 0.75, color: [244, 109, 67] },
        { p: 1, color: [165, 0, 38] },
      ];

  const position =
    max === min
      ? 0.5
      : Math.max(0, Math.min(1, (value - min) / (max - min)));

  let index = 0;

  while (
    index < stops.length - 2 &&
    position > stops[index + 1].p
  ) {
    index++;
  }

  const first = stops[index];
  const second = stops[index + 1];
  const fraction =
    (position - first.p) / (second.p - first.p);

  const rgb = first.color.map((channel, i) =>
    Math.round(
      channel + (second.color[i] - channel) * fraction
    )
  );

  return `rgb(${rgb.join(",")})`;
}

function nearestIndex(values, target) {
  if (!Array.isArray(values) || values.length === 0) {
    return -1;
  }

  let best = 0;

  for (let i = 1; i < values.length; i++) {
    if (
      Math.abs(values[i] - target) <
      Math.abs(values[best] - target)
    ) {
      best = i;
    }
  }

  return best;
}

function getDistrictName(feature) {
  const properties = feature?.properties ?? {};

  return (
    properties.DISTRICT ||
    properties.District ||
    properties.district ||
    properties.NAME_2 ||
    properties.name ||
    properties.dtname ||
    "Maharashtra district"
  );
}

export default function ClimateMap({
  variable = "max_temperature",
  date,
  mode = "historical",
  comparisonSource = "openmeteo",
  showDistricts = true,
}) {
  const [geoData, setGeoData] = useState(null);
  const [grid, setGrid] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  const normalizedVariable = normalizeVariable(variable);
  const isRainfall = normalizedVariable === "rainfall";
  const isMinimum = normalizedVariable === "tmin";

  const title = isRainfall
    ? "Rainfall"
    : isMinimum
      ? "Minimum temperature"
      : "Maximum temperature";

  const units = isRainfall ? "mm" : "°C";

  useEffect(() => {
    let cancelled = false;

    async function loadMap() {
      setLoading(true);
      setError("");
      setGrid(null);

      try {
        // Load Maharashtra district boundaries.
        const boundaryResponse = await fetch(BOUNDARY_URL);

        if (!boundaryResponse.ok) {
          throw new Error(
            "Could not load Maharashtra district boundaries. Check frontend/public/data/maharashtra-districts.geojson."
          );
        }

        const boundary = await boundaryResponse.json();

        let mapGrid;

        // LIVE WEATHER OR FORECAST COMPARISON
        if (mode === "live" || mode === "comparison") {
          const isAIComparison =
            mode === "comparison" &&
            comparisonSource === "convlstm";

          const endpoint = isAIComparison
            ? "comparison-grid"
            : "live-grid";

          const query = new URLSearchParams();

          if (mode === "comparison") {
            query.set("variable", normalizedVariable);

            if (date) {
              query.set("date", date);
            }
          }

          const queryString = query.toString();

          const response = await fetch(
            `${API_BASE_URL}/${endpoint}${
              queryString ? `?${queryString}` : ""
            }`
          );

          if (!response.ok) {
            throw new Error(
              `Climate API returned ${response.status}: ${await response.text()}`
            );
          }

          const payload = await response.json();

          if (isAIComparison) {
            // The ConvLSTM endpoint returns one selected grid directly.
            mapGrid = payload;
          } else {
            // Parse the selected date from the provider's forecast data.
            const selectedDay = payload.dates?.find(
              (item) => item.date === date
            );

            if (!selectedDay) {
              throw new Error(
                `No live weather data is available for ${
                  date || "the selected date"
                }.`
              );
            }

            const layerKey = {
              tmax: "tmax_c",
              tmin: "tmin_c",
              rainfall: "rainfall_mm",
            }[normalizedVariable];

            const selectedLayer =
              selectedDay.layers?.[layerKey];

            if (
              !selectedLayer ||
              !Array.isArray(selectedLayer.grid)
            ) {
              throw new Error(
                `The ${layerKey} layer is missing from the live weather data.`
              );
            }

            mapGrid = {
              variable: normalizedVariable,
              prediction_date: selectedDay.date,
              source: payload.source || "Open-Meteo",
              data_type: selectedDay.data_type,
              disclaimer: payload.disclaimer,
              latitudes: payload.grid?.latitude,
              longitudes: payload.grid?.longitude,
              values: selectedLayer.grid,
            };
          }
        } else {
          // HISTORICAL DATA OR EXISTING SAVED PREDICTIONS
          const query = new URLSearchParams({
            variable: normalizedVariable,
          });

          if (date) {
            query.set("date", date);
          }

          const endpoint =
            mode === "historical"
              ? "historical-grid"
              : "prediction-grid";

          const response = await fetch(
            `${API_BASE_URL}/${endpoint}?${query.toString()}`
          );

          if (!response.ok) {
            throw new Error(
              `Climate data API returned ${response.status}: ${await response.text()}`
            );
          }

          mapGrid = await response.json();
        }

        // VALIDATE GRID COORDINATES AND DIMENSIONS
        if (
          !Array.isArray(mapGrid?.latitudes) ||
          !Array.isArray(mapGrid?.longitudes) ||
          !Array.isArray(mapGrid?.values)
        ) {
          throw new Error(
            "Climate API response is missing grid coordinates or values."
          );
        }

        if (
          mapGrid.values.length !== mapGrid.latitudes.length ||
          mapGrid.values.some(
            (row) =>
              !Array.isArray(row) ||
              row.length !== mapGrid.longitudes.length
          )
        ) {
          throw new Error(
            "Climate grid dimensions do not match the coordinates."
          );
        }

        if (!cancelled) {
          setGeoData(boundary);
          setGrid(mapGrid);
        }
      } catch (err) {
        if (!cancelled) {
          setError(
            err.message || "Unable to load climate map data."
          );
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    loadMap();

    return () => {
      cancelled = true;
    };
  }, [normalizedVariable, date, mode, comparisonSource]);

  const range = useMemo(() => {
    if (!grid?.values) {
      return { min: 0, max: 1 };
    }

    const values = grid.values
      .flat()
      .filter(
        (value) =>
          typeof value === "number" &&
          Number.isFinite(value)
      );

    if (values.length === 0) {
      return { min: 0, max: 1 };
    }

    return {
      min: Math.min(...values),
      max: Math.max(...values),
    };
  }, [grid]);

  function districtValue(feature) {
    if (!grid?.values) return null;

    const bounds = L.geoJSON(feature).getBounds();

    if (!bounds.isValid()) return null;

    const center = bounds.getCenter();

    const row = nearestIndex(grid.latitudes, center.lat);
    const column = nearestIndex(grid.longitudes, center.lng);

    if (row < 0 || column < 0) return null;

    const value = grid.values[row]?.[column];

    return Number.isFinite(value) ? value : null;
  }

  function styleDistrict(feature) {
    const value = districtValue(feature);

    return {
      color: "#ffffff",
      weight: showDistricts ? 1.1 : 0.4,
      opacity: 1,
      fillColor: getColor(
        value,
        range.min,
        range.max,
        isRainfall
      ),
      fillOpacity: value === null ? 0.25 : 0.88,
    };
  }

  function onEachDistrict(feature, layer) {
    const value = districtValue(feature);
    const name = getDistrictName(feature);

    if (value !== null) {
      layer.bindTooltip(
        `<strong>${name}</strong><br/>${title}: ${value.toFixed(1)} ${units}`,
        { sticky: true }
      );
    } else {
      layer.bindTooltip(name, { sticky: true });
    }

    layer.on({
      mouseover: (event) => {
        event.target.setStyle({
          weight: 2,
          color: "#263b36",
          fillOpacity: 1,
        });

        event.target.bringToFront();
      },

      mouseout: (event) => {
        event.target.setStyle(styleDistrict(feature));
      },
    });
  }

  const dataDate =
    grid?.prediction_date || date || "Date unavailable";

  const displayedSource =
    mode === "comparison"
      ? comparisonSource === "convlstm"
        ? "ConvLSTM AI"
        : "Open-Meteo"
      : grid?.source || "Open-Meteo";

  return (
    <div className="leaflet-shell">
      <MapContainer
        center={[19.2, 75.5]}
        zoom={6}
        minZoom={5}
        maxZoom={10}
        scrollWheelZoom
        zoomControl
        className="climate-leaflet"
      >
        <TileLayer
          attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
          url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
        />

        {geoData && grid && (
          <>
            <GeoJSON
              key={`${mode}-${comparisonSource}-${normalizedVariable}-${date}-${showDistricts}`}
              data={geoData}
              style={styleDistrict}
              onEachFeature={onEachDistrict}
            />

            <FitMaharashtra data={geoData} />
          </>
        )}
      </MapContainer>

      {loading && (
        <div className="map-message">
          Loading climate data…
        </div>
      )}

      {!loading && error && (
        <div className="map-message">
          <strong>
            {isRainfall
              ? "Rainfall unavailable"
              : "Unable to load climate map"}
          </strong>
          <span>{error}</span>
        </div>
      )}

      {!loading && !error && grid && (
        <div className="temperature-legend">
          <div className="legend-heading">
            {title} ({units})
          </div>

          <div
            className="legend-gradient"
            style={{
              background: isRainfall
                ? "linear-gradient(to right, #fff7bc, #fec44f, #fe9929, #ec7014, #993404)"
                : "linear-gradient(to right, rgb(49,130,189), rgb(107,174,214), rgb(254,224,139), rgb(244,109,67), rgb(165,0,38))",
            }}
          />

          <div className="legend-values">
            <span>
              {range.min.toFixed(1)} {units}
            </span>
            <span>
              {((range.min + range.max) / 2).toFixed(1)} {units}
            </span>
            <span>
              {range.max.toFixed(1)} {units}
            </span>
          </div>

          <div className="legend-date">
            Data date: {dataDate}

            {(mode === "live" || mode === "comparison") && (
              <>
                <br />
                Source: {displayedSource}
                <br />
                {mode === "comparison" &&
                comparisonSource === "convlstm"
                  ? "Model-generated forecast"
                  : grid.data_type === "forecast"
                    ? "Provider forecast"
                    : "Current-day provider estimate"}
              </>
            )}

            {mode === "comparison" &&
              comparisonSource === "convlstm" &&
              grid.disclaimer && (
                <>
                  <br />
                  {grid.disclaimer}
                </>
              )}
          </div>
        </div>
      )}
    </div>
  );
}
