/**
 * predict.js — Mock sentiment prediction logic
 *
 * Giai đoạn 1: Dùng rule-based keywords tiếng Việt để tạo ra kết quả
 *              có vẻ hợp lý (không hoàn toàn random).
 * Giai đoạn 2: Uncomment phần callAPI() và kết nối endpoint thật.
 */

// ── Keyword dictionaries ──────────────────────────────────────────────────

/*const POSITIVE_KEYWORDS = [
  'tốt', 'giỏi', 'hay', 'xuất sắc', 'tuyệt', 'tuyệt vời', 'hài lòng',
  'nhiệt tình', 'dễ hiểu', 'rõ ràng', 'thú vị', 'hữu ích', 'hiệu quả',
  'tận tâm', 'cảm ơn', 'thích', 'yêu thích', 'bổ ích', 'ấn tượng',
  'chuyên nghiệp', 'sáng tạo', 'truyền đạt tốt', 'dễ tiếp thu',
  'rất hay', 'rất tốt', 'rất nhiệt tình', 'giảng hay', 'giảng tốt',
  'vui', 'thoải mái', 'năng động', 'cuốn hút', 'lôi cuốn', 'phong phú',
];

const NEGATIVE_KEYWORDS = [
  'tệ', 'kém', 'dở', 'chán', 'khó hiểu', 'nhàm chán', 'buồn ngủ',
  'không hiểu', 'không rõ', 'quá khó', 'quá nhanh', 'lơ là',
  'thiếu', 'sai', 'không hài lòng', 'thất vọng', 'bực bội',
  'khó', 'không tốt', 'không hay', 'không nhiệt tình', 'chưa tốt',
  'cần cải thiện', 'chán nản', 'mệt mỏi', 'khó tiếp thu',
  'giảng nhanh', 'giảng chậm quá', 'không rõ ràng', 'giải thích khó',
];

const NEGATION_WORDS = ['không', 'chưa', 'chẳng', 'chả', 'không hề', 'chưa hề'];

const CONTRAST_WORDS = ['nhưng', 'tuy nhiên', 'song', 'dù vậy', 'dù thế', 'tuy', 'mặc dù'];

// ── Core mock prediction ──────────────────────────────────────────────────
*/
/**
 * Phân tích câu tiếng Việt → trả về xác suất 3 class.
 * Logic đơn giản nhưng có xử lý negation và contrast.
 */
/*function mockPredict(text) {
  const lower = text.toLowerCase();
  const tokens = lower.split(/\s+/);

  let posScore = 0;
  let negScore = 0;

  // Keyword matching với negation detection
  for (let i = 0; i < tokens.length; i++) {
    const bigram  = i > 0 ? tokens[i - 1] + ' ' + tokens[i] : '';
    const trigram = i > 1 ? tokens[i - 2] + ' ' + tokens[i - 1] + ' ' + tokens[i] : '';
    const phrase  = trigram || bigram || tokens[i];

    const isNegated = i > 0 && NEGATION_WORDS.some(n => lower.includes(n + ' ' + tokens[i]));

    if (POSITIVE_KEYWORDS.some(k => phrase.includes(k) || tokens[i] === k)) {
      isNegated ? negScore += 1.2 : posScore += 1.5;
    }
    if (NEGATIVE_KEYWORDS.some(k => phrase.includes(k) || tokens[i] === k)) {
      isNegated ? posScore += 0.8 : negScore += 1.5;
    }
  }

  // Contrast reduces confidence of winning side
  const hasContrast = CONTRAST_WORDS.some(w => lower.includes(w));
  if (hasContrast && (posScore > 0 || negScore > 0)) {
    posScore *= 0.7;
    negScore *= 0.7;
  }

  // Add small base noise
  posScore += Math.random() * 0.3;
  negScore += Math.random() * 0.3;
  let neuScore = Math.random() * 0.4;

  // If no strong signals → lean neutral
  const totalSignal = posScore + negScore;
  if (totalSignal < 0.8) neuScore += 1.5;

  // Softmax
  const expPos = Math.exp(posScore);
  const expNeu = Math.exp(neuScore);
  const expNeg = Math.exp(negScore);
  const total  = expPos + expNeu + expNeg;

  return {
    positive: expPos / total,
    neutral:  expNeu / total,
    negative: expNeg / total,
  };
}*/

// ── Future API integration ────────────────────────────────────────────────

/**
 * Giai đoạn 2: Bỏ comment hàm này và gọi thay cho mockPredict().
 * Endpoint FastAPI ví dụ: POST http://localhost:8000/predict
 *
 * async function callAPI(text, model) {
 *   const resp = await fetch('http://localhost:8000/predict', {
 *     method: 'POST',
 *     headers: { 'Content-Type': 'application/json' },
 *     body: JSON.stringify({ text, model }),
 *   });
 *   const data = await resp.json();
 *   return { positive: data.positive, neutral: data.neutral, negative: data.negative };
 * }
 */

  async function callBothAPI(text) {
    const response = await fetch(
      "http://127.0.0.1:8000/predict_both",
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          text: text,
        }),
      }
    );

    if (!response.ok) {
      throw new Error("Backend request failed");
    }

    const data = await response.json();
    if (data.error) {
      throw new Error(data.error);
    }
    return data; // { ann: {...}, lstm: {...} }
  }

// ── UI helpers ────────────────────────────────────────────────────────────

function getLabelInfo(probs) {
  const entries = [
    { key: 'positive', label: 'Tích cực',  emoji: '😊' },
    { key: 'neutral',  label: 'Trung lập', emoji: '😐' },
    { key: 'negative', label: 'Tiêu cực',  emoji: '😞' },
  ];
  return entries.reduce((best, e) =>
    probs[e.key] > probs[best.key] ? e : best, entries[0]
  );
}

function animateBar(el, targetPct) {
  // Small delay so CSS transition triggers properly
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      el.style.width = (targetPct * 100).toFixed(1) + '%';
    });
  });
}

function formatPct(val) {
  return (val * 100).toFixed(1) + '%';
}

// Update one model's verdict badge (label + confidence) from its probs.
function renderVerdict(prefix, probs) {
  const info = getLabelInfo(probs);
  const labelEl = document.getElementById(`${prefix}-label`);
  labelEl.className = `result-label ${info.key}`;
  labelEl.innerHTML = `<span>${info.emoji}</span> ${info.label}`;
  document.getElementById(`${prefix}-confidence`).innerHTML =
    `Độ tin cậy: <strong>${formatPct(probs[info.key])}</strong>`;
}

// ── Comparison chart (3 sentiment classes × 2 models) ──────────────────────

const CHART_VIOLET = '#7c3aed';
const CHART_CYAN   = '#06b6d4';
let compareChart = null;

function renderCompareChart(ann, lstm) {
  const ctx = document.getElementById('chart-compare');
  if (!ctx) return;

  // Class order matches the bar order on the results page: neg / neu / pos
  const annData  = [ann.negative,  ann.neutral,  ann.positive];
  const lstmData = [lstm.negative, lstm.neutral, lstm.positive];

  const config = {
    type: 'bar',
    data: {
      labels: ['😞 Tiêu cực', '😐 Trung lập', '😊 Tích cực'],
      datasets: [
        {
          label: 'ANN + TF-IDF',
          data: annData,
          backgroundColor: 'rgba(124,58,237,0.15)',
          borderColor: CHART_VIOLET,
          borderWidth: 2,
          borderRadius: 6,
          borderSkipped: false,
        },
        {
          label: 'LSTM + Word2Vec',
          data: lstmData,
          backgroundColor: 'rgba(6,182,212,0.15)',
          borderColor: CHART_CYAN,
          borderWidth: 2,
          borderRadius: 6,
          borderSkipped: false,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'top',
          labels: {
            boxWidth: 12, boxHeight: 12, borderRadius: 4,
            usePointStyle: true, pointStyle: 'rectRounded',
            padding: 20, color: '#a09dc0', font: { size: 12, weight: '500' },
          },
        },
        tooltip: {
          backgroundColor: 'rgba(13,11,30,0.95)',
          borderColor: 'rgba(124,58,237,0.3)',
          borderWidth: 1, padding: 12, cornerRadius: 10,
          callbacks: { label: c => ` ${c.dataset.label}: ${(c.parsed.y * 100).toFixed(1)}%` },
        },
      },
      scales: {
        x: {
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#a09dc0', font: { size: 13 } },
        },
        y: {
          min: 0, max: 1,
          grid: { color: 'rgba(255,255,255,0.06)' },
          ticks: { color: '#a09dc0', font: { size: 11 }, callback: v => (v * 100).toFixed(0) + '%' },
        },
      },
      animation: { duration: 700, easing: 'easeOutQuart' },
    },
  };

  // Re-render: destroy old chart so the canvas is reusable across submits.
  if (compareChart) compareChart.destroy();
  compareChart = new Chart(ctx, config);
}

// ── Main predict flow ─────────────────────────────────────────────────────

async function runPrediction() {
  const textarea    = document.getElementById('feedback-input');
  const btn         = document.getElementById('predict-btn');
  const resultPanel = document.getElementById('result-panel');

  const text = textarea.value.trim();
  if (!text) {
    textarea.focus();
    textarea.style.borderColor = 'var(--negative)';
    textarea.style.boxShadow   = '0 0 0 3px rgba(239,68,68,0.15)';
    setTimeout(() => {
      textarea.style.borderColor = '';
      textarea.style.boxShadow   = '';
    }, 1200);
    return;
  }

  // Loading state
  btn.disabled  = true;
  btn.innerHTML = '<div class="spinner"></div> Đang phân tích...';

  try {
    const { ann, lstm } = await callBothAPI(text);

    // Per-model verdict badges
    renderVerdict('ann', ann);
    renderVerdict('lstm', lstm);

    // Grouped comparison chart
    renderCompareChart(ann, lstm);

    // Show panel
    resultPanel.classList.add('visible');
  } catch (err) {
    alert('Lỗi khi gọi backend: ' + err.message);
  } finally {
    // Reset button
    btn.disabled  = false;
    btn.innerHTML = '<span>🔍</span> Phân tích &amp; so sánh';
  }
}

// ── Example sentences ─────────────────────────────────────────────────────

function useExample(text) {
  const textarea = document.getElementById('feedback-input');
  textarea.value = text;
  textarea.focus();
  // Smooth scroll to prediction panel
  document.getElementById('predict-section').scrollIntoView({ behavior: 'smooth', block: 'center' });
}

// ── Init ──────────────────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  // Predict on Ctrl+Enter
  const textarea = document.getElementById('feedback-input');
  if (textarea) {
    textarea.addEventListener('keydown', (e) => {
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        runPrediction();
      }
    });
  }

  // Predict button
  const btn = document.getElementById('predict-btn');
  if (btn) btn.addEventListener('click', runPrediction);
});
