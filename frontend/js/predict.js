/**
 * predict.js — Mock sentiment prediction logic
 *
 * Giai đoạn 1: Dùng rule-based keywords tiếng Việt để tạo ra kết quả
 *              có vẻ hợp lý (không hoàn toàn random).
 * Giai đoạn 2: Uncomment phần callAPI() và kết nối endpoint thật.
 */

// ── Keyword dictionaries ──────────────────────────────────────────────────

const POSITIVE_KEYWORDS = [
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

/**
 * Phân tích câu tiếng Việt → trả về xác suất 3 class.
 * Logic đơn giản nhưng có xử lý negation và contrast.
 */
function mockPredict(text) {
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
}

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

// ── Main predict flow ─────────────────────────────────────────────────────

async function runPrediction() {
  const textarea  = document.getElementById('feedback-input');
  const modelSel  = document.getElementById('model-select');
  const btn       = document.getElementById('predict-btn');
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

  // Simulate network delay (replace with real fetch in giai đoạn 2)
  await new Promise(r => setTimeout(r, 700 + Math.random() * 400));

  const probs = mockPredict(text);
  const modelName = modelSel.value;
  const labelInfo = getLabelInfo(probs);

  // ── Render result ──
  // Label badge
  document.getElementById('result-label').className = `result-label ${labelInfo.key}`;
  document.getElementById('result-label').innerHTML =
    `<span>${labelInfo.emoji}</span> ${labelInfo.label}`;

  // Confidence
  document.getElementById('result-confidence').innerHTML =
    `Độ tin cậy: <strong>${formatPct(probs[labelInfo.key])}</strong>`;

  // Model used
  const modelDisplay = modelName === 'lstm'
    ? 'LSTM + Word2Vec' : 'ANN + TF-IDF';
  document.getElementById('result-model').textContent = modelDisplay;

  // Probability bars
  const barPositive = document.getElementById('bar-positive');
  const barNeutral  = document.getElementById('bar-neutral');
  const barNegative = document.getElementById('bar-negative');

  // Reset to 0 first
  barPositive.style.width = '0%';
  barNeutral.style.width  = '0%';
  barNegative.style.width = '0%';

  animateBar(barPositive, probs.positive);
  animateBar(barNeutral,  probs.neutral);
  animateBar(barNegative, probs.negative);

  document.getElementById('val-positive').textContent = formatPct(probs.positive);
  document.getElementById('val-neutral').textContent  = formatPct(probs.neutral);
  document.getElementById('val-negative').textContent = formatPct(probs.negative);

  // Show panel
  resultPanel.classList.add('visible');

  // Reset button
  btn.disabled  = false;
  btn.innerHTML = '<span>🔍</span> Phân tích cảm xúc';
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
