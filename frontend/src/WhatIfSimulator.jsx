
import { useState } from "react";

const API = "http://127.0.0.1:8000/api";

const initialChanges = {
  tmax_delta: 2,
  tmin_delta: 1,
  rainfall_delta: 10,
};

const fields = [
  {
    key: "tmax_delta",
    label: "Maximum temperature",
    unit: "°C",
    min: -10,
    max: 10,
    step: 0.5,
  },
  {
    key: "tmin_delta",
    label: "Minimum temperature",
    unit: "°C",
    min: -10,
    max: 10,
    step: 0.5,
  },
  {
    key: "rainfall_delta",
    label: "Rainfall",
    unit: "mm/day",
    min: -100,
    max: 100,
    step: 1,
  },
];

function Metric({ label, baseline, scenario, unit }) {
  const difference =
    baseline != null && scenario != null
      ? scenario - baseline
      : null;

  return (
    <div className="whatif-metric">
      <span>{label}</span>
      <div className="whatif-values">
        <div>
          <small>Baseline</small>
          <strong>
            {baseline == null ? "—" : `${baseline.toFixed(2)} ${unit}`}
          </strong>
        </div>
        <div>
          <small>Scenario</small>
          <strong>
            {scenario == null ? "—" : `${scenario.toFixed(2)} ${unit}`}
          </strong>
        </div>
        <div>
          <small>Change</small>
          <strong>
            {difference == null
              ? "—"
              : `${difference > 0 ? "+" : ""}${difference.toFixed(2)} ${unit}`}
          </strong>
        </div>
      </div>
    </div>
  );
}

export default function WhatIfSimulator() {
  const [changes, setChanges] = useState(initialChanges);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function updateChange(key, value) {
    setChanges((previous) => ({
      ...previous,
      [key]: value,
    }));
  }

  async function runScenario(event) {
    event.preventDefault();
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API}/what-if`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          tmax_delta: Number(changes.tmax_delta),
          tmin_delta: Number(changes.tmin_delta),
          rainfall_delta: Number(changes.rainfall_delta),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Could not run the scenario.");
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Unable to connect to the backend. Check that the API is running."
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <section className="whatif-page">
      <div className="whatif-heading">
        <div>
          <p className="whatif-eyebrow">CLIMATE DIGITAL TWIN</p>
          <h2>What-If Climate Simulator</h2>
          <p>
            Explore how hypothetical temperature and rainfall changes affect
            the existing Maharashtra forecast.
          </p>
        </div>
      </div>

      <form className="whatif-controls" onSubmit={runScenario}>
        {fields.map((field) => (
          <label className="whatif-field" key={field.key}>
            <span>{field.label}</span>
            <strong>
              {Number(changes[field.key]) > 0 ? "+" : ""}
              {changes[field.key]} {field.unit}
            </strong>
            <input
              type="range"
              min={field.min}
              max={field.max}
              step={field.step}
              value={changes[field.key]}
              onChange={(event) =>
                updateChange(field.key, event.target.value)
              }
            />
            <small>
              Allowed range: {field.min} to {field.max} {field.unit}
            </small>
          </label>
        ))}

        <div className="whatif-actions">
          <button type="submit" disabled={loading}>
            {loading ? "Running scenario..." : "Run What-If Scenario"}
          </button>
          <button
            type="button"
            className="whatif-reset"
            onClick={() => {
              setChanges(initialChanges);
              setResult(null);
              setError("");
            }}
          >
            Reset
          </button>
        </div>
      </form>

      {error && <p className="whatif-error">{error}</p>}

      {result && (
        <div className="whatif-results">
          <h3>Baseline vs hypothetical scenario</h3>
          <p className="whatif-notice">{result.notice}</p>

          {result.baseline.map((day, index) => {
            const hypothetical = result.scenario[index];

            return (
              <article className="whatif-day" key={day.date}>
                <h4>{day.date}</h4>
                <Metric
                  label="Average maximum temperature"
                  baseline={day.tmax.mean}
                  scenario={hypothetical.tmax.mean}
                  unit="°C"
                />
                <Metric
                  label="Average minimum temperature"
                  baseline={day.tmin.mean}
                  scenario={hypothetical.tmin.mean}
                  unit="°C"
                />
                <Metric
                  label="Average rainfall"
                  baseline={day.rainfall.mean}
                  scenario={hypothetical.rainfall.mean}
                  unit="mm/day"
                />
              </article>
            );
          })}
        </div>
      )}

      <p className="whatif-disclaimer">
        Experimental sensitivity analysis only. This version adjusts forecast
        outputs; it does not rerun the trained ConvLSTM model. Results are not
        official forecasts, alerts, or causal predictions.
      </p>
    </section>
  );
}
