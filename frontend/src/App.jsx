
import { useEffect, useState } from "react";
import {
  CloudSun,
  Map as MapIcon,
  Layers3,
  CalendarDays,
  ChevronDown,
  Droplets,
  Thermometer,
  Info,
  LayoutDashboard,
  ShieldAlert,
  BrainCircuit,
  GitCompareArrows,
  Activity,
  Sprout,
  Waves,
  ArrowUpRight,
  RefreshCw,
  Database,
  Sparkles,
  SlidersHorizontal,
} from "lucide-react";
import ClimateMap from "./components/ClimateMap";
import "./App.css";

import WhatIfSimulator from "./WhatIfSimulator";


const API = "http://127.0.0.1:8000/api";

const layers = [
  {
    id: "max_temperature",
    label: "Maximum temperature",
    unit: "°C",
    icon: Thermometer,
  },
  {
    id: "min_temperature",
    label: "Minimum temperature",
    unit: "°C",
    icon: Thermometer,
  },
  {
    id: "rainfall",
    label: "Rainfall",
    unit: "mm",
    icon: Droplets,
  },
];

const navigation = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "comparison", label: "Forecast comparison", icon: GitCompareArrows },
  { id: "risk", label: "Risk & impact", icon: ShieldAlert },
  { id: "explain", label: "Explainable AI", icon: BrainCircuit },
  { id: "whatif", label: "What-If Simulator", icon: SlidersHorizontal },
];

const modeLabels = {
  historical: "Historical",
  live: "Live weather",
  predictions: "Predictions",
  comparison: "Comparison",
};

function SectionHeading({ eyebrow, title, description }) {
  return (
    <div className="section-heading">
      <span className="eyebrow">{eyebrow}</span>
      <h1>{title}</h1>
      <p>{description}</p>
    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  detail,
  tone = "green",
}) {
  return (
    <article className={`metric-card tone-${tone}`}>
      <div className="metric-top">
        <span className="metric-icon">
          <Icon size={19} />
        </span>
        <ArrowUpRight size={16} className="metric-arrow" />
      </div>
      <span className="metric-label">{label}</span>
      <strong className="metric-value">{value}</strong>
      <span className="metric-detail">{detail}</span>
    </article>
  );
}

export default function App() {
  const [activeView, setActiveView] = useState("overview");
  const [activeLayer, setActiveLayer] = useState("max_temperature");
  const [mode, setMode] = useState("historical");
  const [showDistricts, setShowDistricts] = useState(true);

  // Keep dates separate for each data source.
  const [historicalDates, setHistoricalDates] = useState([]);
  const [predictionDates, setPredictionDates] = useState([]);
  const [liveWeatherDates, setLiveWeatherDates] = useState([]);
  const [comparisonDates, setComparisonDates] = useState([]);

  const [historicalDate, setHistoricalDate] = useState("");
  const [predictionDate, setPredictionDate] = useState("");
  const [liveWeatherDate, setLiveWeatherDate] = useState("");
  const [comparisonDate, setComparisonDate] = useState("");

  const [comparisonSource, setComparisonSource] = useState("openmeteo");

  const [explainabilityData, setExplainabilityData] = useState(null);
  
  const [riskData, setRiskData] = useState(null);
  const [riskStatus, setRiskStatus] = useState("idle");
  const [riskError, setRiskError] = useState("");
  const [riskRefresh, setRiskRefresh] = useState(0);

  const [explainabilityStatus, setExplainabilityStatus] = useState("idle");
  const [explainabilityError, setExplainabilityError] = useState("");
  const [explainabilityRefresh, setExplainabilityRefresh] = useState(0);

  const layer = layers.find((item) => item.id === activeLayer) || layers[0];

  // Always use the date belonging to the currently selected mode.
  const availableDates =
    mode === "historical"
      ? historicalDates
      : mode === "live"
        ? liveWeatherDates
        : mode === "comparison"
          ? comparisonDates
          : predictionDates;

  const date =
    mode === "historical"
      ? historicalDate
      : mode === "live"
        ? liveWeatherDate
        : mode === "comparison"
          ? comparisonDate
          : predictionDate;

  function handleDateChange(nextDate) {
    if (mode === "historical") {
      setHistoricalDate(nextDate);
    } else if (mode === "live") {
      setLiveWeatherDate(nextDate);
    } else if (mode === "comparison") {
      setComparisonDate(nextDate);
    } else {
      setPredictionDate(nextDate);
    }
  }

  function handleModeChange(nextMode) {
    setMode(nextMode);
  }

  // Load date lists independently. One endpoint can never overwrite
  // the selected date belonging to another endpoint.
  useEffect(() => {
    let cancelled = false;

    async function loadEndpoint(endpoint) {
      const response = await fetch(`${API}/${endpoint}`);

      if (!response.ok) {
        throw new Error(
          `${endpoint} returned HTTP ${response.status}`
        );
      }

      return response.json();
    }

    async function loadHistoricalDates() {
      try {
        const data = await loadEndpoint("historical-dates");
        if (cancelled) return;

        const dates = data.dates || [];
        setHistoricalDates(dates);

        setHistoricalDate((current) =>
          dates.includes(current)
            ? current
            : dates[dates.length - 1] || ""
        );
      } catch (error) {
        console.error("Failed to load historical dates:", error);
      }
    }

    async function loadPredictionDates() {
      try {
        const data = await loadEndpoint("prediction-dates");
        if (cancelled) return;

        const dates = data.dates || [];
        setPredictionDates(dates);

        setPredictionDate((current) =>
          dates.includes(current)
            ? current
            : dates[dates.length - 1] || ""
        );
      } catch (error) {
        console.error("Failed to load prediction dates:", error);
      }
    }

    async function loadLiveWeatherDates() {
      try {
        const data = await loadEndpoint("live-grid");
        if (cancelled) return;

        const dates = (data.dates || [])
          .map((item) =>
            typeof item === "string" ? item : item.date
          )
          .filter(Boolean);

        setLiveWeatherDates(dates);

        setLiveWeatherDate((current) =>
          dates.includes(current) ? current : dates[0] || ""
        );
      } catch (error) {
        console.error("Failed to load live weather dates:", error);
      }
    }

    async function loadComparisonDates() {
      try {
        const data = await loadEndpoint("comparison-dates");
        if (cancelled) return;

        const dates = data.dates || [];
        setComparisonDates(dates);

        setComparisonDate((current) =>
          dates.includes(current) ? current : dates[0] || ""
        );
      } catch (error) {
        console.error("Failed to load comparison dates:", error);
      }
    }

    loadHistoricalDates();
    loadPredictionDates();
    loadLiveWeatherDates();
    loadComparisonDates();

    return () => {
      cancelled = true;
    };
  }, []);

  // Load explainability metadata when the Explainable AI section opens.
  useEffect(() => {
    if (activeView !== "explain") return;

    let cancelled = false;

    async function loadExplainability() {
      setExplainabilityStatus("loading");
      setExplainabilityError("");

      try {
        const response = await fetch(
          `${API}/explainability/features`
        );

        if (!response.ok) {
          throw new Error(
            `Explainability endpoint returned HTTP ${response.status}.`
          );
        }

        const data = await response.json();

        if (!cancelled) {
          setExplainabilityData(data);
          setExplainabilityStatus("success");
        }
      } catch (error) {
        if (!cancelled) {
          setExplainabilityError(error.message);
          setExplainabilityStatus("error");
        }
      }
    }

    loadExplainability();

    return () => {
      cancelled = true;
    };
  }, [activeView, explainabilityRefresh]);

  
  useEffect(() => {
    if (activeView !== "risk") return;

    let cancelled = false;

    async function loadRiskIndicators() {
      setRiskStatus("loading");
      setRiskError("");

      try {
        const response = await fetch(`${API}/risk-indicators`);

        if (!response.ok) {
          const message = await response.text();
          throw new Error(
            `Risk API returned HTTP ${response.status}: ${message}`
          );
        }

        const data = await response.json();

        if (!cancelled) {
          setRiskData(data);
          setRiskStatus("success");
        }
      } catch (error) {
        if (!cancelled) {
          setRiskError(error.message || "Unable to load risk indicators.");
          setRiskStatus("error");
        }
      }
    }

    loadRiskIndicators();

    return () => {
      cancelled = true;
    };
  }, [activeView, riskRefresh]);

  return (
    <main className="app-shell">
      <header className="topbar">
        <a
          className="brand"
          href="/"
          aria-label="Maharashtra Climate Atlas home"
        >
          <span className="brand-mark">
            <CloudSun size={25} />
          </span>

          <span className="brand-copy">
            <strong>
              CLIMATE<span> ATLAS</span>
            </strong>
            <small>MAHARASHTRA · INDIA</small>
          </span>
        </a>

        <div className="header-status">
          <span className="status-dot" />
          Climate intelligence workspace
        </div>

        <div className="header-location">
          <span className="location-icon">
            <MapIcon size={15} />
          </span>
          Maharashtra <ChevronDown size={15} />
        </div>
      </header>

      <nav className="main-navigation" aria-label="Dashboard sections">
        {navigation.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            type="button"
            className={`nav-tab ${activeView === id ? "active" : ""}`}
            onClick={() => setActiveView(id)}
            aria-current={activeView === id ? "page" : undefined}
          >
            <Icon size={17} />
            <span>{label}</span>
            {id === "explain" && (
              <span className="nav-tag">AI</span>
            )}
          </button>
        ))}
      </nav>

      <div className="page-content">
        {activeView === "overview" && (
          <>
            <SectionHeading
              eyebrow="MAHARASHTRA / CLIMATE MONITORING"
              title="Understand your climate."
              description="Explore historical conditions, live weather and model-generated forecasts across Maharashtra."
            />

            <div className="welcome-banner">
              <div className="welcome-symbol">
                <CloudSun size={26} />
              </div>

              <div className="welcome-copy">
                <span>YOUR REGIONAL CLIMATE VIEW</span>
                <strong>
                  Explore conditions, one district at a time.
                </strong>
                <p>
                  Choose a data mode, variable and date to update the map.
                </p>
              </div>

              <div className="banner-decoration">
                <Activity size={72} />
              </div>
            </div>

            <div className="workspace">
              <aside className="control-panel">
                <div className="panel-title">
                  <span className="panel-title-icon">
                    <Layers3 size={17} />
                  </span>
                  <div>
                    <h2>Map controls</h2>
                    <p>Customize your climate view</p>
                  </div>
                </div>

                <div className="control-group">
                  <label className="control-label">
                    01 / DATA SOURCE
                  </label>

                  <div className="mode-grid">
                    {[
                      ["historical", "Historical", "Past observations"],
                      ["live", "Live weather", "Provider data"],
                      ["predictions", "Predictions", "Saved forecasts"],
                      ["comparison", "Compare", "AI vs provider"],
                    ].map(([id, title, sub]) => (
                      <button
                        key={id}
                        className={`mode-card ${
                          mode === id ? "selected" : ""
                        }`}
                        onClick={() => handleModeChange(id)}
                        aria-pressed={mode === id}
                        type="button"
                      >
                        <span className="mode-card-title">
                          {title}
                        </span>
                        <small>{sub}</small>
                      </button>
                    ))}
                  </div>
                </div>

                {mode === "comparison" && (
                  <div className="control-group">
                    <label className="control-label">
                      FORECAST SOURCE
                    </label>

                    <div className="source-switch">
                      <button
                        className={
                          comparisonSource === "openmeteo"
                            ? "selected"
                            : ""
                        }
                        onClick={() =>
                          setComparisonSource("openmeteo")
                        }
                        type="button"
                      >
                        Open-Meteo
                      </button>

                      <button
                        className={
                          comparisonSource === "convlstm"
                            ? "selected"
                            : ""
                        }
                        onClick={() =>
                          setComparisonSource("convlstm")
                        }
                        type="button"
                      >
                        ConvLSTM AI
                      </button>
                    </div>
                  </div>
                )}

                <div className="control-group">
                  <label className="control-label">
                    02 / CLIMATE VARIABLE
                  </label>

                  <div className="variable-list">
                    {layers.map(
                      ({ id, label, unit, icon: Icon }) => (
                        <button
                          key={id}
                          className={`variable-option ${
                            activeLayer === id ? "selected" : ""
                          }`}
                          onClick={() => setActiveLayer(id)}
                          aria-pressed={activeLayer === id}
                          type="button"
                        >
                          <span className="variable-icon">
                            <Icon size={17} />
                          </span>

                          <span className="variable-copy">
                            <strong>{label}</strong>
                            <small>{unit}</small>
                          </span>

                          <span className="variable-radio" />
                        </button>
                      )
                    )}
                  </div>
                </div>

                <div className="control-group">
                  <label
                    className="control-label"
                    htmlFor="climate-date"
                  >
                    03 / DATE
                  </label>

                  <div className="date-input-wrap">
                    <CalendarDays size={17} />

                    <select
                      id="climate-date"
                      value={date}
                      onChange={(event) =>
                        handleDateChange(event.target.value)
                      }
                      disabled={!availableDates.length}
                    >
                      {availableDates.map((item) => (
                        <option key={item} value={item}>
                          {item}
                        </option>
                      ))}
                    </select>
                  </div>

                  {!availableDates.length && (
                    <p className="inline-help">
                      No dates are available for this data source.
                      Check the backend API and its data files.
                    </p>
                  )}
                </div>

                <div className="district-toggle">
                  <span className="toggle-icon">
                    <Layers3 size={17} />
                  </span>

                  <span className="toggle-text">
                    <strong>District boundaries</strong>
                    <small>Administrative outlines</small>
                  </span>

                  <button
                    type="button"
                    role="switch"
                    aria-checked={showDistricts}
                    aria-label="Toggle district boundaries"
                    className={`switch ${
                      showDistricts ? "on" : ""
                    }`}
                    onClick={() =>
                      setShowDistricts((value) => !value)
                    }
                  >
                    <span />
                  </button>
                </div>

                <div className="panel-note">
                  <Info size={15} />
                  <span>
                    Available map coverage depends on the selected
                    data source and the backend datasets.
                  </span>
                </div>
              </aside>

              <section className="map-section">
                <div className="map-toolbar">
                  <div>
                    <span className="map-kicker">
                      SPATIAL DISTRIBUTION
                    </span>

                    <h2>{layer.label}</h2>

                    <p>
                      {mode === "comparison"
                        ? `${
                            comparisonSource === "convlstm"
                              ? "ConvLSTM AI"
                              : "Open-Meteo"
                          } forecast`
                        : mode === "live"
                          ? "Live weather data"
                          : mode === "predictions"
                            ? "Saved model predictions"
                            : "Historical climate data"}
                      {date ? ` · ${date}` : ""}
                    </p>
                  </div>

                  <span className="unit-chip">{layer.unit}</span>
                </div>

                <div className="map-frame">
                  {date ? (
                    <ClimateMap
                      variable={activeLayer}
                      date={date}
                      mode={mode}
                      comparisonSource={comparisonSource}
                      showDistricts={showDistricts}
                    />
                  ) : (
                    <div className="map-empty-state">
                      <CalendarDays size={28} />
                      <h3>Waiting for available dates</h3>
                      <p>
                        The map will load when this data source
                        provides a valid date.
                      </p>
                    </div>
                  )}

                  <div className="map-stamp">
                    <span className="status-dot" />
                    MAHARASHTRA · INDIA
                  </div>
                </div>

                <div className="map-footer">
                  <div className="legend-title">
                    <span
                      className={`legend-swatch ${
                        activeLayer === "rainfall" ? "rain" : ""
                      }`}
                    />
                    <span>
                      <strong>{layer.label}</strong>
                      <small>Map colour scale · {layer.unit}</small>
                    </span>
                  </div>

                  <span className="footer-note">
                    Interactive district map
                  </span>
                </div>
              </section>
            </div>

            <div className="overview-cards">
              <MetricCard
                icon={Thermometer}
                label="Climate variable"
                value={layer.label}
                detail={`Displayed in ${layer.unit}`}
              />

              <MetricCard
                icon={CalendarDays}
                label="Selected date"
                value={date || "Not available"}
                detail={`Selected ${modeLabels[mode].toLowerCase()} date`}
                tone="sand"
              />

              <MetricCard
                icon={Database}
                label="Data mode"
                value={modeLabels[mode]}
                detail="Switch modes in Map controls"
                tone="blue"
              />
            </div>
          </>
        )}

        {activeView === "comparison" && (
          <>
            <SectionHeading
              eyebrow="MODEL EVALUATION / FORECASTS"
              title="Compare forecast sources."
              description="View the same variable and date using Open-Meteo or your ConvLSTM AI forecast."
            />

            <div className="comparison-banner">
              <div className="welcome-symbol">
                <GitCompareArrows size={25} />
              </div>

              <div className="welcome-copy">
                <span>FORECAST COMPARISON</span>
                <strong>
                  Compare maps using identical settings.
                </strong>
                <p>
                  Switch between sources to inspect their spatial
                  patterns. This view does not itself calculate
                  model accuracy.
                </p>
              </div>
            </div>

            <div className="comparison-controls">
              <div className="field-block">
                <label htmlFor="comparison-variable">
                  Climate variable
                </label>

                <select
                  id="comparison-variable"
                  value={activeLayer}
                  onChange={(event) =>
                    setActiveLayer(event.target.value)
                  }
                >
                  {layers.map((item) => (
                    <option key={item.id} value={item.id}>
                      {item.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="field-block">
                <label htmlFor="comparison-date">
                  Forecast date
                </label>

                <select
                  id="comparison-date"
                  value={comparisonDate}
                  onChange={(event) =>
                    setComparisonDate(event.target.value)
                  }
                  disabled={!comparisonDates.length}
                >
                  {comparisonDates.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>

                {!comparisonDates.length && (
                  <small className="inline-help">
                    No comparison dates are available.
                  </small>
                )}
              </div>

              <div className="field-block">
                <label>Source</label>

                <div className="source-switch">
                  <button
                    className={
                      comparisonSource === "openmeteo"
                        ? "selected"
                        : ""
                    }
                    onClick={() =>
                      setComparisonSource("openmeteo")
                    }
                    type="button"
                  >
                    Open-Meteo
                  </button>

                  <button
                    className={
                      comparisonSource === "convlstm"
                        ? "selected"
                        : ""
                    }
                    onClick={() =>
                      setComparisonSource("convlstm")
                    }
                    type="button"
                  >
                    ConvLSTM AI
                  </button>
                </div>
              </div>

              <button
                className="primary-button"
                onClick={() => {
                  setMode("comparison");
                  setActiveView("overview");
                }}
                disabled={!comparisonDate}
                type="button"
              >
                <MapIcon size={16} />
                View on map
              </button>
            </div>

            <div className="info-grid">
              <article className="info-card">
                <span className="info-card-icon green">
                  <Layers3 size={20} />
                </span>
                <h3>Same variable</h3>
                <p>
                  Keep maximum temperature, minimum temperature or
                  rainfall selected for a consistent visual comparison.
                </p>
              </article>

              <article className="info-card">
                <span className="info-card-icon blue">
                  <CalendarDays size={20} />
                </span>
                <h3>Same forecast date</h3>
                <p>
                  Only dates supplied by the comparison forecast
                  endpoint are selectable.
                </p>
              </article>

              <article className="info-card">
                <span className="info-card-icon gold">
                  <Activity size={20} />
                </span>
                <h3>Accuracy evaluation</h3>
                <p>
                  Quantitative metrics such as MAE and RMSE require
                  aligned predictions and reference observations.
                </p>
              </article>
            </div>
          </>
        )}

        {activeView === "risk" && (
          <>
            <SectionHeading
              eyebrow="DECISION SUPPORT / CLIMATE RESILIENCE"
              title="Risk & impact assessment."
              description="A dedicated workspace for interpreting climate hazards and turning validated forecasts into practical guidance."
            />

            <div className="risk-banner">
              <div className="welcome-symbol">
                <ShieldAlert size={25} />
              </div>

              <div className="welcome-copy">
                <span>RISK ASSESSMENT WORKSPACE</span>
                <strong>
                  From climate signals to informed action.
                </strong>
                  
                  <p>
                    {riskStatus === "loading" &&
                      "Loading forecast-derived climate indicators…"}
                    {riskStatus === "error" &&
                      "Risk indicators could not be loaded. Check the API and retry."}
                    {riskStatus === "success" &&
                      "Experimental temperature and rainfall indicators are connected to the forecast API."}
                    {riskStatus === "idle" &&
                      "Open this section to load forecast-derived climate indicators."}
                  </p>

      
              </div>

              
            <span className="preview-pill">
              {riskStatus === "loading"
                ? "LOADING"
                : riskStatus === "success"
                  ? "EXPERIMENTAL DATA"
                  : riskStatus === "error"
                    ? "API ERROR"
                    : "READY"}
            </span>

            </div>


            <div className="risk-results">
              <div className="section-subheading">
                <div>
                  <span className="eyebrow">
                    MODEL OUTPUT / DAILY INDICATORS
                  </span>
                  <h2>Forecast risk indicators</h2>
                  <p>
                    Daily summaries across valid Maharashtra grid cells.
                  </p>
                </div>

                <button
                  type="button"
                  className="risk-refresh"
                  onClick={() => setRiskRefresh((value) => value + 1)}
                  disabled={riskStatus === "loading"}
                >
                  <RefreshCw size={16} />
                  Refresh
                </button>
              </div>

              {riskStatus === "loading" && (
                <p className="state-message">
                  Loading risk indicators…
                </p>
              )}

              {riskStatus === "error" && (
                <div className="state-message state-error">
                  <p>{riskError}</p>
                  <button
                    type="button"
                    onClick={() => setRiskRefresh((value) => value + 1)}
                  >
                    Try again
                  </button>
                </div>
              )}

              {riskStatus === "success" && riskData && (
                <>
                  <div className="risk-data-meta">
                    <span>
                      Source: {riskData.source_file}
                    </span>
                    <span>
                      Valid grid cells: {riskData.valid_spatial_cells}
                    </span>
                    <span>
                      Forecast days: {riskData.results?.length ?? 0}
                    </span>
                  </div>

                  <div className="risk-day-list">
                    {(riskData.results || []).map((day) => (
                      <article
                        className="risk-day-card"
                        key={day.date}
                      >
                        <div className="risk-day-heading">
                          <h3>{day.date}</h3>
                          <span>Experimental forecast</span>
                        </div>

                        <div className="risk-day-metrics">
                          <div>
                            <span>Mean Tmax</span>
                            <strong>
                              {day.temperature.tmax.status === "available"
                                ? `${Number(day.temperature.tmax.mean).toFixed(1)} °C`
                                : "N/A"}
                            </strong>
                          </div>

                          <div>
                            <span>Mean Tmin</span>
                            <strong>
                              {day.temperature.tmin.status === "available"
                                ? `${Number(day.temperature.tmin.mean).toFixed(1)} °C`
                                : "N/A"}
                            </strong>
                          </div>

                          <div>
                            <span>Mean rainfall</span>
                            <strong>
                              {day.rainfall.status === "available"
                                ? `${Number(day.rainfall.mean).toFixed(1)} mm`
                                : "N/A"}
                            </strong>
                          </div>
                        </div>

                        <div className="risk-rainfall-counts">
                          <strong>Rainfall threshold indicators</strong>
                          <span>
                            Heavy:{" "}
                            {day.rainfall.threshold_indicators
                              .heavy_64_5_to_115_5_mm} grid cells
                          </span>
                          <span>
                            Very heavy:{" "}
                            {day.rainfall.threshold_indicators
                              .very_heavy_115_6_to_204_4_mm} grid cells
                          </span>
                          <span>
                            Extremely heavy:{" "}
                            {day.rainfall.threshold_indicators
                              .extremely_heavy_204_5_mm_or_more} grid cells
                          </span>
                        </div>

                        <p className="risk-day-note">
                          Heatwave classification: not assessed.
                          Rainfall counts represent grid cells, not districts.
                        </p>
                      </article>
                    ))}
                  </div>

                  <p className="disclaimer-note">
                    <Info size={16} />
                    {riskData.notice}
                  </p>
                </>
              )}
            </div>


            <div className="risk-grid">
              <article className="risk-card risk-heat">
                <div className="risk-card-top">
                  <span className="risk-icon">
                    <Thermometer size={21} />
                  </span>
                  <span className="risk-type">HEAT HAZARD</span>
                </div>
                <h2>Heat stress</h2>
                <p>
                  Assess forecast maximum temperatures against
                  documented heat thresholds.
                </p>
                <div className="risk-card-foot">
                  <span>Required input</span>
                  <strong>Maximum temperature</strong>
                </div>
              </article>

              <article className="risk-card risk-rain">
                <div className="risk-card-top">
                  <span className="risk-icon">
                    <Waves size={21} />
                  </span>
                  <span className="risk-type">
                    RAINFALL HAZARD
                  </span>
                </div>
                <h2>Heavy rainfall</h2>
                <p>
                  Identify potential rainfall concerns using
                  validated daily rainfall thresholds.
                </p>
                <div className="risk-card-foot">
                  <span>Required input</span>
                  <strong>Rainfall forecast</strong>
                </div>
              </article>

              <article className="risk-card risk-dry">
                <div className="risk-card-top">
                  <span className="risk-icon">
                    <Sprout size={21} />
                  </span>
                  <span className="risk-type">AGRICULTURE</span>
                </div>
                <h2>Dry-spell concern</h2>
                <p>
                  Assess sustained low rainfall alongside available
                  agricultural context.
                </p>
                <div className="risk-card-foot">
                  <span>Required input</span>
                  <strong>Rainfall time series</strong>
                </div>
              </article>
            </div>

            <div className="section-subheading">
              <div>
                <span className="eyebrow">RESPONSE PLANNING</span>
                <h2>Recommendation categories</h2>
              </div>
            </div>

            <div className="recommendation-list">
              <article>
                <span className="recommendation-icon">
                  <Droplets size={19} />
                </span>
                <div>
                  <h3>Water management</h3>
                  <p>
                    Potential actions: review irrigation schedules,
                    soil moisture and local water availability when
                    supported by verified conditions.
                  </p>
                </div>
                <span className="recommendation-label">
                  Guidance framework
                </span>
              </article>

              <article>
                <span className="recommendation-icon">
                  <ShieldAlert size={19} />
                </span>
                <div>
                  <h3>Heat preparedness</h3>
                  <p>
                    Potential actions: consult official heat advisories
                    and local public-health guidance when forecast
                    thresholds are exceeded.
                  </p>
                </div>
                <span className="recommendation-label">
                  Guidance framework
                </span>
              </article>

              <article>
                <span className="recommendation-icon">
                  <Sprout size={19} />
                </span>
                <div>
                  <h3>Crop planning</h3>
                  <p>
                    Potential actions: combine forecast conditions
                    with crop stage and local agricultural advisories
                    before making decisions.
                  </p>
                </div>
                <span className="recommendation-label">
                  Guidance framework
                </span>
              </article>
            </div>

            <p className="disclaimer-note">
              <Info size={16} />
              No live risk score or district-specific recommendation
              is calculated on this screen yet. Guidance is general,
              not an official warning.
            </p>
          </>
        )}

    {activeView === "whatif" && <WhatIfSimulator />}
    {activeView === "explain" && (
      <>
        <SectionHeading
          eyebrow="MODEL TRANSPARENCY / INTERPRETABILITY"
          title="Explainable AI"
          description="Explore which climate variables and historical input days contribute most to the ConvLSTM model's temperature predictions."
        />

        <div className="explain-banner">
          <div className="explain-orbit">
            <BrainCircuit size={32} />
          </div>
          <div className="welcome-copy">
            <span>SHAP-BASED MODEL EXPLANATION</span>
            <strong>Understand what influences the forecast.</strong>
            <p>
              These global importance values summarize model attribution.
              They do not establish causation or explain every individual grid cell.
            </p>
          </div>
          <span className="ai-chip">
            <Sparkles size={14} /> XAI
          </span>
        </div>

        {explainabilityStatus === "loading" && (
          <p className="state-message">Loading feature importance...</p>
        )}

        {explainabilityStatus === "error" && (
          <p className="state-message state-error">{explainabilityError}</p>
        )}

        {explainabilityStatus === "success" && explainabilityData && (
          <>
            <div className="info-grid">
              <article className="info-card">
                <span className="info-card-icon green">
                  <Database size={20} />
                </span>
                <h3>10 climate variables</h3>
                <p>Inputs include IMD, ERA5, IMR and PET data.</p>
              </article>
              <article className="info-card">
                <span className="info-card-icon blue">
                  <BrainCircuit size={20} />
                </span>
                <h3>SHAP GradientExplainer</h3>
                <p>Attribution outputs generated for Tmax and Tmin.</p>
              </article>
              <article className="info-card">
                <span className="info-card-icon gold">
                  <Activity size={20} />
                </span>
                <h3>7 input days</h3>
                <p>The model uses a seven-day historical input window.</p>
              </article>
            </div>

            {[
              { key: "tmax", label: "Maximum temperature (Tmax)" },
              { key: "tmin", label: "Minimum temperature (Tmin)" },
            ].map((target) => {
              const features = Object.entries(
                explainabilityData[target.key] || {}
              ).sort((a, b) => b[1] - a[1]);

              return (
                <section className="data-panel" key={target.key}>
                  <div className="data-panel-heading">
                    <div>
                      <span className="eyebrow">GLOBAL FEATURE IMPORTANCE</span>
                      <h2>{target.label}</h2>
                    </div>
                  </div>

                  {features.map(([name, value]) => (
                    <div key={name} style={{ marginBottom: 17 }}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          gap: 12,
                          marginBottom: 6,
                          fontSize: 13,
                        }}
                      >
                        <span>{name}</span>
                        <strong>{Number(value).toFixed(2)}%</strong>
                      </div>
                      <div
                        style={{
                          height: 10,
                          borderRadius: 8,
                          background: "var(--border-color, #e5e7eb)",
                          overflow: "hidden",
                        }}
                      >
                        <div
                          style={{
                            height: "100%",
                            width: `${Math.max(0, Math.min(100, Number(value)))}%`,
                            background:
                              target.key === "tmax" ? "#0d9488" : "#6366f1",
                            borderRadius: 8,
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </section>
              );
            })}

            <section className="data-panel">
              <div className="data-panel-heading">
                <div>
                  <span className="eyebrow">EXISTING SHAP OUTPUTS</span>
                  <h2>Feature and temporal importance plots</h2>
                </div>
              </div>

              <div className="info-grid">
                {[
                  ["tmax_feature_importance.png", "Tmax feature importance"],
                  ["tmin_feature_importance.png", "Tmin feature importance"],
                  ["tmax_temporal_importance.png", "Tmax temporal importance"],
                  ["tmin_temporal_importance.png", "Tmin temporal importance"],
                ].map(([file, title]) => (
                  <article className="info-card" key={file}>
                    <h3>{title}</h3>
                    <img
                      src={`/data/explainability/${file}`}
                      alt={title}
                      loading="lazy"
                      style={{
                        width: "100%",
                        height: "auto",
                        display: "block",
                        borderRadius: 10,
                        marginTop: 12,
                      }}
                      onError={(event) => {
                        event.currentTarget.style.display = "none";
                      }}
                    />
                  </article>
                ))}
              </div>
            </section>

            <section className="data-panel">
              <div className="data-panel-heading">
                <div>
                  <span className="eyebrow">SPATIAL ATTRIBUTION</span>
                  <h2>Spatial SHAP visualizations</h2>
                </div>
              </div>
              <div className="info-grid">
                {[
                  ["tmax_spatial_shap.png", "Tmax spatial attribution"],
                  ["tmin_spatial_shap.png", "Tmin spatial attribution"],
                ].map(([file, title]) => (
                  <article className="info-card" key={file}>
                    <h3>{title}</h3>
                    <img
                      src={`/data/explainability/${file}`}
                      alt={title}
                      loading="lazy"
                      style={{
                        width: "100%",
                        height: "auto",
                        display: "block",
                        borderRadius: 10,
                        marginTop: 12,
                      }}
                      onError={(event) => {
                        event.currentTarget.style.display = "none";
                      }}
                    />
                  </article>
                ))}
              </div>
            </section>

            <p className="disclaimer-note">
              <Info size={16} />
              Global feature importance describes attribution in the current
              explanation run. It is not a causal effect, a forecast confidence
              score, or a substitute for model validation.
            </p>
          </>
        )}
      </>
    )}
    
      </div>

      <footer className="app-footer">
        <span>
          <CloudSun size={15} />
          MAHARASHTRA CLIMATE ATLAS
        </span>

        <span>
          AI-assisted regional climate exploration
          <span className="footer-separator"> • </span>
          Decision support prototype
        </span>
      </footer>
    </main>
  );
}
