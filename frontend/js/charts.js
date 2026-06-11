/**
 * charts.js — Chart.js visualizations for the Results Dashboard
 *
 * Uses hardcoded experiment results from the README / reports.
 * All data sourced from multi-seed evaluation (mean ± std).
 */

// ── Experiment data ───────────────────────────────────────────────────────

const VALIDATION_DATA = {
  labels: ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'],
  ann: [0.8929, 0.7483, 0.8208, 0.7706],
  lstm: [0.8932, 0.7586, 0.8216, 0.7807],
};

const TEST_DATA = {
  labels: ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'],
  ann: [0.8628, 0.7137, 0.7629, 0.7282],
  lstm: [0.8707, 0.7273, 0.7868, 0.7444],
};

const IMPROVEMENT_DATA = {
  labels: ['Accuracy', 'Macro Precision', 'Macro Recall', 'Macro F1'],
  values: [0.0079, 0.0136, 0.0239, 0.0162],
};

// ── Chart.js default config ───────────────────────────────────────────────

Chart.defaults.color = '#a09dc0';
Chart.defaults.font.family = "'Inter', sans-serif";
Chart.defaults.font.size = 12;

const GRID_COLOR = 'rgba(255,255,255,0.06)';
const VIOLET = '#7c3aed';
const CYAN   = '#06b6d4';
const VIOLET_BG = 'rgba(124,58,237,0.15)';
const CYAN_BG   = 'rgba(6,182,212,0.15)';

function baseBarOptions(yMin = 0.6, yMax = 1.0) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          boxWidth: 12,
          boxHeight: 12,
          borderRadius: 4,
          usePointStyle: true,
          pointStyle: 'rectRounded',
          padding: 20,
          color: '#a09dc0',
          font: { size: 12, weight: '500' },
        },
      },
      tooltip: {
        backgroundColor: 'rgba(13,11,30,0.95)',
        borderColor: 'rgba(124,58,237,0.3)',
        borderWidth: 1,
        padding: 12,
        cornerRadius: 10,
        callbacks: {
          label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(4)}`,
        },
      },
    },
    scales: {
      x: {
        grid: { color: GRID_COLOR, drawBorder: false },
        ticks: { color: '#a09dc0', font: { size: 12 } },
      },
      y: {
        min: yMin,
        max: yMax,
        grid: { color: GRID_COLOR, drawBorder: false },
        ticks: {
          color: '#a09dc0',
          font: { size: 11 },
          callback: v => v.toFixed(2),
        },
      },
    },
    animation: {
      duration: 900,
      easing: 'easeOutQuart',
    },
    layout: { padding: { top: 8 } },
  };
}

// ── Render: Validation comparison ────────────────────────────────────────

function renderValidationChart() {
  const ctx = document.getElementById('chart-validation');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: VALIDATION_DATA.labels,
      datasets: [
        {
          label: 'ANN + TF-IDF',
          data: VALIDATION_DATA.ann,
          backgroundColor: VIOLET_BG,
          borderColor: VIOLET,
          borderWidth: 2,
          borderRadius: 6,
          borderSkipped: false,
        },
        {
          label: 'LSTM + Word2Vec',
          data: VALIDATION_DATA.lstm,
          backgroundColor: CYAN_BG,
          borderColor: CYAN,
          borderWidth: 2,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: baseBarOptions(0.7, 0.93),
  });
}

// ── Render: Test comparison ────────────────────────────────────────────────

function renderTestChart() {
  const ctx = document.getElementById('chart-test');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: TEST_DATA.labels,
      datasets: [
        {
          label: 'ANN + TF-IDF',
          data: TEST_DATA.ann,
          backgroundColor: VIOLET_BG,
          borderColor: VIOLET,
          borderWidth: 2,
          borderRadius: 6,
          borderSkipped: false,
        },
        {
          label: 'LSTM + Word2Vec',
          data: TEST_DATA.lstm,
          backgroundColor: CYAN_BG,
          borderColor: CYAN,
          borderWidth: 2,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: baseBarOptions(0.65, 0.92),
  });
}

// ── Render: Improvement horizontal bar ────────────────────────────────────

function renderImprovementChart() {
  const ctx = document.getElementById('chart-improvement');
  if (!ctx) return;

  new Chart(ctx, {
    type: 'bar',
    data: {
      labels: IMPROVEMENT_DATA.labels,
      datasets: [
        {
          label: 'Cải thiện (LSTM vs ANN)',
          data: IMPROVEMENT_DATA.values,
          backgroundColor: [
            'rgba(16,185,129,0.25)',
            'rgba(16,185,129,0.35)',
            'rgba(16,185,129,0.45)',
            'rgba(16,185,129,0.55)',
          ],
          borderColor: '#10b981',
          borderWidth: 2,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: {
      indexAxis: 'y',
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: 'rgba(13,11,30,0.95)',
          borderColor: 'rgba(16,185,129,0.3)',
          borderWidth: 1,
          padding: 12,
          cornerRadius: 10,
          callbacks: {
            label: ctx => ` +${ctx.parsed.x.toFixed(4)}`,
          },
        },
      },
      scales: {
        x: {
          min: 0,
          grid: { color: GRID_COLOR },
          ticks: {
            color: '#a09dc0',
            callback: v => '+' + v.toFixed(3),
          },
        },
        y: {
          grid: { display: false },
          ticks: { color: '#a09dc0' },
        },
      },
      animation: { duration: 900, easing: 'easeOutQuart' },
    },
  });
}

// ── Init ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  renderValidationChart();
  renderTestChart();
  renderImprovementChart();
});
