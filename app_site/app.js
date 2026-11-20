import { prepareSeries, predictAt, addBusinessDays, parseCSV, mergeObservations, forecastCSV, computeMetrics, RAW_COLUMNS } from './engine.js';

const $ = selector => document.querySelector(selector);
const $$ = selector => [...document.querySelectorAll(selector)];
const number = (value, decimals = 2) => Number.isFinite(value) ? new Intl.NumberFormat('es-ES', { minimumFractionDigits: decimals, maximumFractionDigits: decimals }).format(value) : '—';
const shortDate = value => new Intl.DateTimeFormat('es-ES', { day: '2-digit', month: 'short', year: 'numeric', timeZone: 'UTC' }).format(new Date(`${value.slice(0, 10)}T00:00:00Z`));
const escapeHTML = value => String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[char]));
const dateMillis = value => Date.parse(`${String(value).slice(0, 10)}T00:00:00Z`);
const dateISO = value => new Date(value).toISOString().slice(0, 10);
const COLORS = { actual: '#b7cbed', prediction: '#52d1be', naive: '#e6b775', future: '#baace8' };
const STORAGE_KEY = 'brent-observatorio-added-observations-v1';
const MIN_ANNUAL_OBSERVATIONS = 30;
const state = { tab: 'pronostico', range: '1a', horizon: 5, evaluationHorizon: 5, evaluationYear: 'all', contextEvent: null, importedCount: 0 };
let model, data, originalData, evaluation, prepared, forecast;

function safeURL(url) { try { const parsed = new URL(url); return ['https:', 'http:'].includes(parsed.protocol) ? parsed.href : '#'; } catch { return '#'; } }
function text(selector, value) { $(selector).textContent = value; }
function download(contents, filename, type = 'text/csv;charset=utf-8') {
  const url = URL.createObjectURL(new Blob([contents], { type }));
  const link = document.createElement('a'); link.href = url; link.download = filename; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function csv(rows) { return '\uFEFF' + rows.map(row => row.map(value => `"${String(value ?? '').replaceAll('"', '""')}"`).join(',')).join('\r\n') + '\r\n'; }
function resizeThrottle(callback) { let timer; return () => { clearTimeout(timer); timer = setTimeout(callback, 140); }; }
function switchTab(value, writeHash = false) {
  if (!['pronostico', 'evaluacion', 'contexto', 'datos'].includes(value)) value = 'pronostico';
  state.tab = value;
  $$('.view').forEach(view => { view.hidden = view.id !== value; view.classList.toggle('active', view.id === value); });
  $$('[data-tab]').forEach(link => { const active = link.dataset.tab === value; link.classList.toggle('active', active); if (active) link.setAttribute('aria-current', 'page'); else link.removeAttribute('aria-current'); });
  text('#breadcrumb-current', ({ pronostico: 'Pronóstico', evaluacion: 'Evaluación', contexto: 'Contexto', datos: 'Datos y modelo' })[value]);
  if (writeHash) history.replaceState(null, '', `#${value}`);
  if (value === 'pronostico') renderForecastChart();
  if (value === 'evaluacion') renderEvaluation();
  if (value === 'contexto') renderContext();
}

/** SVG chart with real date coordinates, keyboard-readable forecast points and no smoothed prices. */
function lineChart(containerSelector, { series = [], forecastPoints = [], origin = null, originBase = null, futureReference = null, minDate = null, maxDate = null, annotations = [], readout = null, title = '' } = {}) {
  const container = $(containerSelector);
  const allPoints = [...series.flatMap(line => line.points), ...forecastPoints.map(point => ({ date: point.targetDate, value: point.price }))].filter(point => Number.isFinite(point.value));
  container.replaceChildren();
  if (!allPoints.length) { const empty = document.createElement('div'); empty.className = 'empty-chart'; empty.textContent = 'No hay observaciones disponibles para esta selección.'; container.append(empty); return; }
  const width = Math.max(container.clientWidth || 700, 320), height = Math.max(container.clientHeight || 330, 270);
  const margin = { top: 29, right: width < 500 ? 12 : 25, bottom: 36, left: width < 500 ? 44 : 53 };
  const innerWidth = width - margin.left - margin.right;
  let xMin = minDate ? dateMillis(minDate) : Math.min(...allPoints.map(point => dateMillis(point.date)));
  let xMax = maxDate ? dateMillis(maxDate) : Math.max(...allPoints.map(point => dateMillis(point.date)));
  if (xMin === xMax) xMax += 86_400_000;
  const x = date => margin.left + (dateMillis(date) - xMin) / (xMax - xMin) * innerWidth;
  // Compact year badges occupy separate lanes above the plot. Full titles remain
  // available on focus/hover and in the episode cards; labels never cover prices.
  const laneEnds = [];
  const eventMarkers = annotations.filter(event => dateMillis(event.date) >= xMin && dateMillis(event.date) <= xMax)
    .sort((a, b) => a.date.localeCompare(b.date)).map(event => {
      const year = event.date.slice(0, 4), endYear = event.endDate?.slice(0, 4);
      const label = endYear && endYear !== year ? `${year}–${endYear}` : year;
      const badgeWidth = label.length * 8 + 18;
      const left = Math.min(width - margin.right - badgeWidth, Math.max(margin.left, x(event.date) - badgeWidth / 2));
      let lane = laneEnds.findIndex(end => end + 8 <= left);
      if (lane < 0) lane = laneEnds.length;
      laneEnds[lane] = left + badgeWidth;
      return { event, label, left, badgeWidth, lane };
    });
  if (eventMarkers.length) margin.top = 29 + laneEnds.length * 28;
  const innerHeight = height - margin.top - margin.bottom;
  const inRange = allPoints.filter(point => dateMillis(point.date) >= xMin && dateMillis(point.date) <= xMax);
  const extrema = [...inRange.map(point => point.value), ...forecastPoints.flatMap(point => [point.lower, point.upper]).filter(Number.isFinite)];
  if (Number.isFinite(originBase)) extrema.push(originBase);
  if (Number.isFinite(futureReference)) extrema.push(futureReference);
  let yMin = Math.min(...extrema), yMax = Math.max(...extrema);
  const padding = Math.max((yMax - yMin) * .14, 3); yMin -= padding; yMax += padding;
  const y = value => margin.top + (yMax - value) / (yMax - yMin) * innerHeight;
  const NS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(NS, 'svg'); svg.setAttribute('viewBox', `0 0 ${width} ${height}`); svg.setAttribute('role', 'img'); svg.setAttribute('aria-label', title);
  const label = document.createElementNS(NS, 'title'); label.textContent = title; svg.append(label);
  const append = (tag, attrs, content, parent = svg) => { const el = document.createElementNS(NS, tag); Object.entries(attrs).forEach(([name, value]) => el.setAttribute(name, value)); if (content !== undefined) el.textContent = content; parent.append(el); return el; };
  const defs = append('defs', {}); const clipID = `clip-${containerSelector.replace(/[^a-z0-9]/gi, '')}`;
  const clip = append('clipPath', { id: clipID }, undefined, defs); append('rect', { x: margin.left, y: margin.top - 10, width: innerWidth, height: innerHeight + 10 }, undefined, clip);
  const plot = append('g', { 'clip-path': `url(#${clipID})` });
  append('text', { x: margin.left, y: 13, class: 'chart-label' }, 'USD / barril');
  for (let step = 0; step <= 4; step++) {
    const value = yMin + (yMax - yMin) * step / 4, yy = y(value);
    append('line', { x1: margin.left, x2: width - margin.right, y1: yy, y2: yy, stroke: '#243b4d', 'stroke-width': 1 });
    append('text', { x: margin.left - 10, y: yy + 4, 'text-anchor': 'end', class: 'chart-axis' }, number(value, 0));
  }
  const tickCount = width < 500 ? 3 : 5;
  for (let index = 0; index <= tickCount; index++) {
    const millis = xMin + (xMax - xMin) * index / tickCount, date = new Date(millis);
    const long = xMax - xMin > 86400000 * 1050;
    const label = long ? String(date.getUTCFullYear()) : new Intl.DateTimeFormat('es-ES', { month: 'short', year: '2-digit', timeZone: 'UTC' }).format(date);
    const xx = margin.left + innerWidth * index / tickCount;
    append('text', { x: xx, y: height - 11, 'text-anchor': index === 0 ? 'start' : index === tickCount ? 'end' : 'middle', class: 'chart-axis' }, label);
  }
  if (origin) {
    const originX = x(origin);
    append('rect', { x: originX, y: margin.top - 4, width: Math.max(0, width - margin.right - originX), height: innerHeight + 4, fill: '#0d302f' }, undefined, plot);
    append('line', { x1: originX, x2: originX, y1: margin.top, y2: height - margin.bottom, stroke: '#547b83', 'stroke-dasharray': '3 4' }, undefined, plot);
    if (originX < width - 95) append('text', { x: Math.max(margin.left, originX + 9), y: margin.top - 9, class: 'chart-date-badge' }, 'PREVISIÓN');
    for (const [value, color] of [[originBase, COLORS.naive], [futureReference, COLORS.future]]) {
      if (Number.isFinite(value)) append('line', { x1: originX, x2: width - margin.right, y1: y(value), y2: y(value), stroke: color, 'stroke-width': 1.4, 'stroke-dasharray': '5 5', opacity: .85 }, undefined, plot);
    }
  }
  eventMarkers.forEach(({ event, label, left, badgeWidth, lane }) => {
    const xx = x(event.date), top = 23 + lane * 28;
    append('line', { x1: xx, x2: xx, y1: top + 24, y2: height - margin.bottom, stroke: '#53697b', 'stroke-dasharray': '3 4', opacity: .8, class: 'chart-annotation-line' });
    const group = append('g', { tabindex: '0', role: 'button', class: 'chart-event-marker', 'aria-label': `${event.title}. Ampliar este episodio.` });
    append('title', {}, event.title, group);
    append('rect', { x: left, y: top, width: badgeWidth, height: 24, rx: 6 }, undefined, group);
    append('text', { x: left + badgeWidth / 2, y: top + 16.5, 'text-anchor': 'middle', class: 'chart-event-label' }, label, group);
    const selectEvent = () => { state.contextEvent = event.id; renderContext(); };
    group.addEventListener('click', selectEvent);
    group.addEventListener('keydown', key => { if (key.key === 'Enter' || key.key === ' ') { key.preventDefault(); selectEvent(); } });
    group.addEventListener('mouseenter', () => showTooltip(escapeHTML(event.title), left, top + 28));
    group.addEventListener('mouseleave', () => hideTooltip());
    group.addEventListener('focus', () => showTooltip(escapeHTML(event.title), left, top + 28));
    group.addEventListener('blur', () => hideTooltip());
  });
  series.forEach(line => {
    let path = '', lastDate = null;
    const visible = line.points.filter(point => Number.isFinite(point.value) && dateMillis(point.date) >= xMin && dateMillis(point.date) <= xMax);
    visible.forEach(point => {
      // Gaps are shown as gaps when more than ten calendar days separate publications.
      const move = lastDate === null || dateMillis(point.date) - lastDate > 86400000 * 10;
      path += `${move ? 'M' : 'L'}${x(point.date).toFixed(2)},${y(point.value).toFixed(2)} `;
      lastDate = dateMillis(point.date);
    });
    append('path', { d: path, fill: 'none', stroke: line.color, 'stroke-width': line.width || 1.8, 'stroke-linecap': 'round', 'stroke-linejoin': 'round', ...(line.dash ? { 'stroke-dasharray': '5 4' } : {}) }, undefined, plot);
  });
  const tooltip = document.createElement('div'); tooltip.className = 'chart-tooltip hidden'; container.append(svg, tooltip);
  const showTooltip = (html, xx, yy) => { tooltip.innerHTML = html; tooltip.classList.remove('hidden'); tooltip.style.left = `${Math.max(2, Math.min(width - 195, xx + 12))}px`; tooltip.style.top = `${Math.max(0, Math.min(height - 115, yy - 60))}px`; };
  const hideTooltip = () => tooltip.classList.add('hidden');
  forecastPoints.forEach(point => {
    const xx = x(point.targetDate), yy = y(point.price), selected = point.horizon === state.horizon;
    if (Number.isFinite(point.lower) && Number.isFinite(point.upper)) {
      const bandWidth = selected ? 27 : 18;
      append('rect', { x: xx - bandWidth / 2, y: y(point.upper), width: bandWidth, height: y(point.lower) - y(point.upper), rx: 4, fill: selected ? '#28695c' : '#1e4c47', opacity: .75 }, undefined, plot);
      append('line', { x1: xx, x2: xx, y1: y(point.upper), y2: y(point.lower), stroke: '#65bca9', 'stroke-width': 1 }, undefined, plot);
      for (const bound of [point.lower, point.upper]) append('line', { x1: xx - 6, x2: xx + 6, y1: y(bound), y2: y(bound), stroke: '#65bca9' }, undefined, plot);
    }
    const group = append('g', { tabindex: '0', role: 'button', class: 'chart-point', 'aria-label': `${point.horizon} días, ${shortDate(point.targetDate)}, ${number(point.price)} dólares por barril. Seleccionar horizonte.` });
    append('circle', { cx: xx, cy: yy, r: 12, fill: '#205a50', class: 'svg-focus-dot' }, undefined, group);
    append('circle', { cx: xx, cy: yy, r: selected ? 6 : 4.5, fill: COLORS.prediction, stroke: '#102331', 'stroke-width': 2 }, undefined, group);
    append('text', { x: Math.min(width - margin.right - 5, Math.max(margin.left + 16, xx)), y: yy - 14, 'text-anchor': 'middle', class: 'chart-date-badge' }, `${point.horizon} d`);
    const content = `<div class="tooltip-date">${escapeHTML(shortDate(point.targetDate))}</div><strong>${point.horizon} días · ${number(point.price)} USD</strong><br>Rango: ${number(point.lower)}–${number(point.upper)}`;
    group.addEventListener('mouseenter', () => showTooltip(content, xx, yy)); group.addEventListener('mouseleave', hideTooltip);
    group.addEventListener('focus', () => showTooltip(content, xx, yy)); group.addEventListener('blur', hideTooltip);
    const select = () => { state.horizon = point.horizon; renderForecast(); };
    group.addEventListener('click', select); group.addEventListener('keydown', event => { if (event.key === 'Enter' || event.key === ' ') { event.preventDefault(); select(); } });
  });
  // Mouse crosshair follows a real stored observation; it never interpolates prices.
  if (series[0]?.points.length) {
    const overlay = append('rect', { x: margin.left, y: margin.top, width: Math.max(1, (origin ? x(origin) : width - margin.right) - margin.left), height: innerHeight, class: 'chart-hit' });
    const crosshair = append('line', { y1: margin.top, y2: height - margin.bottom, stroke: '#8cb9c4', 'stroke-dasharray': '3 3', visibility: 'hidden', 'pointer-events': 'none' });
    const circle = append('circle', { r: 4, fill: series[0].color, stroke: '#102331', 'stroke-width': 2, visibility: 'hidden', 'pointer-events': 'none' });
    const primary = series[0].points.filter(point => Number.isFinite(point.value) && dateMillis(point.date) >= xMin && dateMillis(point.date) <= xMax);
    overlay.addEventListener('mousemove', event => {
      if (!primary.length) return;
      const rect = svg.getBoundingClientRect(), px = (event.clientX - rect.left) * width / rect.width;
      const target = xMin + (px - margin.left) / innerWidth * (xMax - xMin);
      let low = 0, high = primary.length - 1;
      while (low < high) { const middle = Math.floor((low + high) / 2); if (dateMillis(primary[middle].date) < target) low = middle + 1; else high = middle; }
      let point = primary[low]; if (low && Math.abs(dateMillis(primary[low - 1].date) - target) < Math.abs(dateMillis(point.date) - target)) point = primary[low - 1];
      const xx = x(point.date), yy = y(point.value); crosshair.setAttribute('x1', xx); crosshair.setAttribute('x2', xx); crosshair.setAttribute('visibility', 'visible'); circle.setAttribute('cx', xx); circle.setAttribute('cy', yy); circle.setAttribute('visibility', 'visible');
      let content = `<div class="tooltip-date">${escapeHTML(shortDate(point.date))}</div>`;
      const values = [];
      series.forEach(line => { const value = line.points.find(item => item.date === point.date)?.value; if (Number.isFinite(value)) { content += `${escapeHTML(line.name)}: <strong>${number(value)}</strong><br>`; values.push(`${line.name}: ${number(value)} USD`); } });
      showTooltip(content, xx, yy);
      if (readout) text(readout, `${shortDate(point.date)} · ${values.join(' · ')}`);
    });
    overlay.addEventListener('mouseleave', () => { crosshair.setAttribute('visibility', 'hidden'); circle.setAttribute('visibility', 'hidden'); hideTooltip(); });
  }
}

function refreshPrepared() {
  prepared = prepareSeries(data);
  const minimum = prepared.firstUsableIndex + model.window - 1;
  if (minimum < model.window - 1 || minimum >= prepared.dates.length) throw new Error('La instantánea no contiene suficiente historial completo para la red.');
  $('#origin-date').min = prepared.dates[minimum]; $('#origin-date').max = prepared.dates.at(-1);
  $('#origin-date').value = prepared.dates.at(-1);
  forecast = predictAt(model, prepared, $('#origin-date').value);
  text('#origin-note', `Entradas disponibles hasta ${shortDate(prepared.dates.at(-1))}`);
}
function renderForecast() {
  text('#known-price', number(forecast.base)); text('#future-price', `${number(forecast.futureReference)} USD`);
  text('#volatility', `${number(forecast.volatility.annualized * 100, 1)} %`);
  $('#forecast-cards').innerHTML = forecast.predictions.map(point => `<button class="forecast-card ${point.horizon === state.horizon ? 'selected' : ''}" data-horizon="${point.horizon}" aria-pressed="${point.horizon === state.horizon}"><div class="card-top"><span class="horizon-label">${point.horizon} días</span><span class="horizon-tag">${({ 5: 'Corto plazo', 63: 'Medio plazo', 252: 'Largo plazo' })[point.horizon]}</span></div><div class="forecast-value">${number(point.price)}<span>USD / barril</span></div><div class="forecast-delta ${point.change > 0 ? 'up' : point.change < 0 ? 'down' : ''}">${point.change >= 0 ? '+' : ''}${number(point.change)} USD · ${point.changePct >= 0 ? '+' : ''}${number(point.changePct)} %</div><div class="forecast-target">Objetivo: ${escapeHTML(shortDate(point.targetDate))}</div>${point.horizon === state.horizon ? '<span class="selected-indicator" aria-hidden="true"></span>' : ''}</button>`).join('');
  $$('[data-horizon]').forEach(button => button.addEventListener('click', () => { state.horizon = Number(button.dataset.horizon); renderForecast(); }));
  const selected = forecast.predictions.find(point => point.horizon === state.horizon);
  const half = Number(model.intervalHalfWidths?.[String(state.horizon)]);
  const direction = selected.change > .01 ? 'subida' : selected.change < -.01 ? 'bajada' : 'estabilidad';
  text('#interpretation-title', `${state.horizon} días: una señal de ${direction}`);
  text('#interpretation-text', `Con la información conocida el ${shortDate(forecast.origin)}, la red estima ${number(selected.price)} USD por barril para el ${shortDate(selected.targetDate)}. Son ${selected.change >= 0 ? '+' : ''}${number(selected.changePct)} % respecto al Brent conocido en el origen. El rango empírico va de ${number(selected.price - half)} a ${number(selected.price + half)} USD.`);
  const reliability = $('#reliability-note'); reliability.classList.toggle('caution', state.horizon !== 5);
  reliability.textContent = ({ 5: 'En el test 2023–2024, el RMSE fue aproximadamente un 11,4 % menor que el del ingenuo spot. La ventaja no se mantuvo frente a la referencia de futuros.', 63: 'En el test 2023–2024, prácticamente empató con el ingenuo spot. Este horizonte no demuestra una ventaja predictiva estable.', 252: 'En el test 2023–2024, quedó ligeramente por detrás del ingenuo spot. El horizonte largo acumula incertidumbre y las predicciones se solapan.' })[state.horizon];
  text('#chart-readout', `${state.horizon} días · objetivo ${shortDate(selected.targetDate)} · previsión ${number(selected.price)} USD · rango ${number(selected.price - half)}–${number(selected.price + half)} USD.`);
  renderForecastChart();
}
function renderForecastChart() {
  if (!forecast || state.tab !== 'pronostico') return;
  const start = new Date(`${forecast.origin}T00:00:00Z`);
  if (state.range !== 'todo') start.setUTCMonth(start.getUTCMonth() - ({ '3m': 3, '1a': 12, '3a': 36 })[state.range]);
  const minimum = state.range === 'todo' ? data.observed[0].date : dateISO(start);
  const history = data.observed.filter(row => row.date >= minimum && row.date <= forecast.origin).map(row => ({ date: row.date, value: row.price }));
  const points = forecast.predictions.map(point => ({ ...point, lower: point.price - Number(model.intervalHalfWidths[String(point.horizon)]), upper: point.price + Number(model.intervalHalfWidths[String(point.horizon)]) }));
  const knownSpot = prepared.dates.map((date, index) => ({ date, value: prepared.rawRows[index][0] })).filter(point => point.date >= minimum && point.date <= forecast.origin && Number.isFinite(point.value));
  lineChart('#forecast-chart', { title: 'Precio Brent observado, spot conocido y tres previsiones independientes con rangos empíricos', series: [{ name: 'Brent observado', points: history, color: COLORS.actual, width: 2 }, { name: 'Spot conocido', points: knownSpot, color: COLORS.naive, width: 1.2, dash: true }], forecastPoints: points, origin: forecast.origin, originBase: forecast.base, futureReference: forecast.futureReference, minDate: minimum, maxDate: addBusinessDays(points.at(-1).targetDate, 8), readout: '#chart-readout' });
}
function selectedEvaluationRows() { return evaluation.predictions.filter(row => Number(row.horizon) === state.evaluationHorizon && (state.evaluationYear === 'all' || row.target_date.slice(0, 4) === state.evaluationYear)).sort((a, b) => a.target_date.localeCompare(b.target_date)); }
function syncEvaluationYears() {
  const select = $('#evaluation-year'), note = $('#evaluation-sample-note');
  const counts = new Map();
  evaluation.predictions.filter(row => Number(row.horizon) === state.evaluationHorizon).forEach(row => {
    const year = row.target_date.slice(0, 4); counts.set(year, (counts.get(year) || 0) + 1);
  });
  const small = [];
  [...select.options].forEach(option => {
    if (option.value === 'all') return;
    const count = counts.get(option.value) || 0;
    option.disabled = count < MIN_ANNUAL_OBSERVATIONS;
    option.textContent = option.disabled ? `${option.value} (${count} datos)` : option.value;
    if (option.disabled) small.push({ year: option.value, count });
  });
  if (state.evaluationYear !== 'all' && (counts.get(state.evaluationYear) || 0) < MIN_ANNUAL_OBSERVATIONS) state.evaluationYear = 'all';
  select.value = state.evaluationYear;
  note.hidden = small.length === 0;
  note.textContent = small.map(({ year, count }) => `${year} solo aporta ${count} fechas objetivo publicadas a ${state.evaluationHorizon} días.`).join(' ')
    + (state.evaluationHorizon === 252 && small.some(item => item.year === '2023') ? ' Los pronósticos comienzan en enero de 2023 y sus primeras fechas objetivo llegan en diciembre.' : '')
    + ` Esas observaciones siguen incluidas en 2023–2024 y en su CSV. Para evitar resumir periodos muy cortos, la vista anual requiere al menos ${MIN_ANNUAL_OBSERVATIONS} observaciones; este criterio no garantiza independencia estadística.`;
}
function renderEvaluation() {
  syncEvaluationYears();
  const rows = selectedEvaluationRows(), metrics = computeMetrics(rows);
  const metricData = [ ['MAE', metrics.mae, 'USD / barril', 2], ['MSE', metrics.mse, 'USD² / barril²', 2], ['RMSE', metrics.rmse, 'USD / barril', 2], ['R²', metrics.r2, 'Coeficiente de determinación', 3], ['Ratio vs. ingenuo', metrics.relative_rmse, 'Menos de 1 mejora el RMSE', 3], ['Ratio vs. futuros', metrics.relative_rmse_vs_future, 'Referencia conocida al origen', 3] ];
  $('#evaluation-metrics').innerHTML = metricData.map(([name, value, unit, decimals], index) => `<div class="metric-box ${index === 4 ? value < 1 ? 'emphasis' : 'caution' : ''}"><span>${name}</span><strong>${number(value, decimals)}</strong><small>${unit}</small></div>`).join('');
  const coverageRows = rows.filter(row => Number.isFinite(row.lower) && Number.isFinite(row.upper));
  const coverage = coverageRows.length ? coverageRows.filter(row => row.actual >= row.lower && row.actual <= row.upper).length / coverageRows.length * 100 : null;
  text('#evaluation-chart-subtitle', `${state.evaluationHorizon} días · ${number(rows.length, 0)} observaciones · ${state.evaluationYear === 'all' ? 'fechas objetivo 2023–2024' : `fechas objetivo ${state.evaluationYear}`}`);
  const series = [['Real', 'actual', COLORS.actual, false], ['RNN con señales', 'prediction', COLORS.prediction, false], ['Ingenuo spot', 'naive', COLORS.naive, true], ['Referencia de futuros', 'future_naive', COLORS.future, true]].map(([name, key, color, dash]) => ({ name, color, dash, width: key === 'actual' ? 2.2 : 1.6, points: rows.map(row => ({ date: row.target_date, value: row[key] })) }));
  lineChart('#evaluation-chart', { title: `Test histórico a ${state.evaluationHorizon} días: real, RNN y ambas referencias`, series, readout: '#evaluation-readout' });
  const summary = !rows.length ? 'No hay etiquetas reales disponibles para esta combinación de horizonte y año.' : `El ratio frente al ingenuo spot es ${number(metrics.relative_rmse, 3)}: ${metrics.relative_rmse < .995 ? `el RMSE se reduce un ${number((1 - metrics.relative_rmse) * 100, 1)} %` : metrics.relative_rmse > 1.005 ? `el RMSE aumenta un ${number((metrics.relative_rmse - 1) * 100, 1)} %` : 'el resultado es prácticamente un empate'}. ${metrics.relative_rmse_vs_future >= 1 ? 'No supera la referencia de futuros.' : 'Supera la referencia de futuros en este subconjunto.'}${Number.isFinite(coverage) ? ` La cobertura observada del rango empírico es ${number(coverage, 1)} %.` : ''}${metrics.r2 < 0 ? ' El R² negativo obliga a interpretar con cautela la capacidad explicativa.' : ''}`;
  text('#evaluation-insight', summary); text('#evaluation-readout', rows.length ? `${shortDate(rows[0].target_date)}–${shortDate(rows.at(-1).target_date)} · ${rows.length} objetivos con precio real publicado.` : 'Sin objetivos disponibles.');
}
function renderComparison() {
  const groups = new Map(); evaluation.comparison.forEach(row => { const key = `${row.architecture}|${row.variant}`; if (!groups.has(key)) groups.set(key, new Map()); groups.get(key).set(Number(row.horizon), row); });
  $('#comparison-body').innerHTML = [...groups.entries()].map(([key, horizons]) => { const [architecture, variant] = key.split('|'); const chosen = architecture === 'RNN' && variant === 'signals'; return `<tr class="${chosen ? 'chosen' : ''}"><td>${escapeHTML(architecture)} · ${variant === 'signals' ? 'señales' : 'niveles'}${chosen ? '<span class="chosen-tag">Elegido</span>' : ''}</td>${[5, 63, 252].map(horizon => { const ratio = horizons.get(horizon)?.relative_rmse; return `<td class="${ratio < 1 ? 'better' : ''}">${number(ratio, 3)}</td>`; }).join('')}</tr>`; }).join('');
}
function renderContext() {
  if (!data || state.tab !== 'contexto') return;
  const event = data.events.find(item => item.id === state.contextEvent);
  let minDate = null, maxDate = null;
  if (event) { const before = dateMillis(event.date) - 86400000 * 180, after = dateMillis(event.endDate || event.date) + 86400000 * 180; minDate = dateISO(before); maxDate = dateISO(Math.min(after, dateMillis(data.observed.at(-1).date))); }
  lineChart('#context-chart', { title: 'Historia del precio Brent y episodios de contexto económico', series: [{ name: 'Brent observado', color: COLORS.actual, width: 1.8, points: data.observed.map(row => ({ date: row.date, value: row.price })) }], minDate, maxDate, annotations: event ? [event] : data.events, readout: '#context-readout' });
  $$('.event-card').forEach(card => card.classList.toggle('selected', card.dataset.event === state.contextEvent));
  text('#context-readout', event ? `${event.title} · ${shortDate(event.date)}${event.endDate ? `–${shortDate(event.endDate)}` : ''}. El gráfico muestra observaciones, sin atribuir cada movimiento a una única causa.` : `Historia disponible: ${shortDate(data.observed[0].date)}–${shortDate(data.observed.at(-1).date)}. Selecciona un episodio para ampliar el intervalo.`);
}
function renderEvents() {
  $('#event-cards').innerHTML = data.events.map(event => `<article class="event-card" data-event="${escapeHTML(event.id)}"><div class="event-year">${escapeHTML(event.date.slice(0, 4))}${event.endDate && event.endDate.slice(0, 4) !== event.date.slice(0, 4) ? `–${escapeHTML(event.endDate.slice(0, 4))}` : ''}</div><h3>${escapeHTML(event.title)}</h3><p>${escapeHTML(event.description)}</p><button class="button button-text event-select" data-event="${escapeHTML(event.id)}">Explorar este episodio</button>${event.sourceUrl ? `<a class="event-source" href="${escapeHTML(safeURL(event.sourceUrl))}" target="_blank" rel="noopener noreferrer">Fuente</a>` : ''}</article>`).join('');
  $$('.event-select').forEach(button => button.addEventListener('click', () => { state.contextEvent = button.dataset.event; renderContext(); $('#context-chart').scrollIntoView({ behavior: matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'center' }); }));
}
function renderData() {
  const minimum = prepared.dates[prepared.firstUsableIndex + model.window - 1];
  $('#dataset-summary').innerHTML = [['Última entrada disponible', shortDate(prepared.dates.at(-1))], ['Último precio real publicado', shortDate(data.snapshot.lastObservedDate || data.observed.at(-1).date)], ['Primer origen completo', shortDate(minimum)], ['Filas de entradas', number(data.rows.length, 0)], ['Filas añadidas en este navegador', number(state.importedCount, 0)]].map(([label, value]) => `<div><dt>${label}</dt><dd>${value}</dd></div>`).join('');
  const meta = model.metadata || {};
  $('#model-facts').innerHTML = [['RNN simple', `${meta.hiddenSize || 32} unidades`, 'Una capa recurrente'], ['Ventana', `${model.window} días`, 'Información conocida'], ['Parámetros', number(meta.parameters || 1699, 0), `Semilla ${meta.seed ?? 42}`], ['Entrenamiento', `${meta.selectedEpoch || 17} épocas`, 'Pesos originales'], ['Salidas', model.horizons.join(' · '), 'Días de lunes a viernes']].map(([label, value, extra]) => `<div class="model-fact"><span>${label}</span><strong>${value}</strong><small>${extra}</small></div>`).join('');
  text('#model-technical-extra', `Entrenamiento hasta 2017, selección con validación 2018–2022 y evaluación histórica 2023–2024. Dropout 0 y penalización L2 0 en esta versión original. Los rangos se calibraron con errores de validación, no con el test. ${Array.isArray(meta.limitations) ? meta.limitations.join(' ') : ''}`);
  $('#source-list').innerHTML = data.sources.map(source => `<div class="source-item"><a href="${escapeHTML(safeURL(source.url))}" target="_blank" rel="noopener noreferrer">${escapeHTML(source.name)}</a><p>${escapeHTML(source.description)}</p></div>`).join('');
  const snapshotDate = data.snapshot.lastInputDate || originalData.rows.at(-1)[0];
  $('#snapshot-chip').innerHTML = `<span class="snapshot-icon" aria-hidden="true">◷</span> Instantánea · ${escapeHTML(shortDate(snapshotDate))}`;
}
function feedback(message, isError = false) { const node = $('#import-feedback'); node.textContent = message; node.classList.toggle('error', isError); }
async function importFile(file) {
  if (!file) return;
  if (file.size > 10 * 1024 * 1024) { feedback('El fichero supera 10 MB. Importa como máximo 10 000 observaciones.', true); return; }
  try {
    const imported = parseCSV(await file.text());
    const today = new Date().toISOString().slice(0, 10);
    const futureObservation = imported.rows.find(row => row[0] > today);
    if (futureObservation) throw new Error(`La observación del ${shortDate(futureObservation[0])} está en el futuro. Solo se pueden añadir entradas ya publicadas, con fecha hasta ${shortDate(today)} (UTC). Las fechas futuras se reservan para las previsiones.`);
    const merged = mergeObservations(data, imported);
    const nextData = { ...data, ...merged.data };
    const nextPrepared = prepareSeries(nextData);
    predictAt(model, nextPrepared, nextPrepared.dates.at(-1));
    data = nextData; state.importedCount = data.rows.filter(row => row[0] > originalData.rows.at(-1)[0]).length;
    let saved = true;
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify({ columns: data.columns, rows: data.rows.filter(row => row[0] > originalData.rows.at(-1)[0]) })); } catch { saved = false; }
    refreshPrepared(); renderForecast(); renderData();
    feedback(`${merged.added} observaciones añadidas; ${merged.unchanged} ya existentes.${merged.filled ? ` Se completaron ${merged.filled} días intermedios con la última entrada conocida, exclusivamente como entradas del modelo.` : ''} La nueva fecha de origen es ${shortDate(prepared.dates.at(-1))}. ${saved ? 'Los cambios se guardaron en este navegador.' : 'Los cambios están disponibles en esta sesión; no se pudo usar el almacenamiento local.'} Los precios reales de evaluación permanecen separados.`);
  } catch (error) { feedback(error.message || 'No se ha podido leer el CSV.', true); }
  $('#csv-file').value = '';
}
function bindEvents() {
  $$('a[href^="#"]').forEach(link => link.addEventListener('click', event => {
    const tab = link.getAttribute('href').slice(1);
    if (!['pronostico', 'evaluacion', 'contexto', 'datos'].includes(tab)) return;
    event.preventDefault(); switchTab(tab, true); window.scrollTo({ top: 0, behavior: 'instant' });
  }));
  window.addEventListener('hashchange', () => switchTab(location.hash.slice(1)));
  window.addEventListener('resize', resizeThrottle(() => { if (state.tab === 'pronostico') renderForecastChart(); if (state.tab === 'evaluacion') renderEvaluation(); if (state.tab === 'contexto') renderContext(); }));
  $('#origin-date').addEventListener('change', event => {
    const previous = forecast.origin;
    try { forecast = predictAt(model, prepared, event.target.value); renderForecast(); text('#origin-note', 'Solo se usa información disponible hasta este origen.'); }
    catch (error) { event.target.value = previous; text('#origin-note', error.message); }
  });
  $$('[data-range]').forEach(button => button.addEventListener('click', () => { state.range = button.dataset.range; $$('[data-range]').forEach(item => item.classList.toggle('selected', item === button)); renderForecastChart(); }));
  $('#export-forecast').addEventListener('click', () => download('\uFEFF' + forecastCSV(forecast, model), `brent-prevision-${forecast.origin}.csv`));
  $('#evaluation-horizon').addEventListener('change', event => { state.evaluationHorizon = Number(event.target.value); renderEvaluation(); });
  $('#evaluation-year').addEventListener('change', event => { state.evaluationYear = event.target.value; renderEvaluation(); });
  $('#export-evaluation').addEventListener('click', () => { const rows = selectedEvaluationRows(); const keys = ['horizon', 'origin', 'target_date', 'actual', 'prediction', 'naive', 'future_naive', 'lower', 'upper']; download(csv([keys, ...rows.map(row => keys.map(key => row[key]))]), `brent-test-${state.evaluationHorizon}d-${state.evaluationYear}.csv`); });
  $('#context-reset').addEventListener('click', () => { state.contextEvent = null; renderContext(); });
  $('#csv-file').addEventListener('change', event => importFile(event.target.files[0]));
  const zone = $('#dropzone'); ['dragenter', 'dragover'].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.add('dragging'); })); ['dragleave', 'drop'].forEach(name => zone.addEventListener(name, event => { event.preventDefault(); zone.classList.remove('dragging'); })); zone.addEventListener('drop', event => importFile(event.dataTransfer.files[0]));
  $('#download-template').addEventListener('click', () => { const last = data.rows.at(-1); download(csv([['Date', ...RAW_COLUMNS], [addBusinessDays(last[0], 1), ...RAW_COLUMNS.map(() => '')]]), 'plantilla-actualizacion-brent.csv'); feedback('Plantilla descargada. Sustituye las celdas vacías por datos ya publicados. Una celda vacía conserva la última entrada conocida; no representa un precio real nuevo.'); });
  $('#download-data').addEventListener('click', () => download(csv([data.columns, ...data.rows]), `brent-entradas-${prepared.dates.at(-1)}.csv`));
  $('#restore-data').addEventListener('click', () => { data = structuredClone(originalData); state.importedCount = 0; try { localStorage.removeItem(STORAGE_KEY); } catch { /* Storage may be disabled in a local file. */ } refreshPrepared(); renderForecast(); renderData(); feedback('Instantánea original restaurada. Se han eliminado únicamente las filas añadidas en este navegador.'); });
}
async function start() {
  try {
    const bundle = window.__BRENT_BUNDLE__ || await (async () => { const paths = ['model.json', 'app-data.json', 'evaluation.json']; const result = await Promise.all(paths.map(async path => { const response = await fetch(`./data/${path}`); if (!response.ok) throw new Error(`No se pudo cargar ${path} (${response.status}).`); return response.json(); })); return { model: result[0], data: result[1], evaluation: result[2] }; })();
    model = bundle.model; data = bundle.data; evaluation = bundle.evaluation;
    if (!model || !data?.rows?.length || !data.observed?.length || !evaluation?.predictions || !evaluation?.comparison) throw new Error('La instantánea está incompleta; faltan el modelo, los datos o sus resultados históricos.');
    originalData = structuredClone(data);
    let restoredMessage = '';
    try { const saved = localStorage.getItem(STORAGE_KEY); if (saved) { const imported = JSON.parse(saved); const merged = mergeObservations(data, imported); data = { ...data, ...merged.data }; state.importedCount = data.rows.length - originalData.rows.length; restoredMessage = `Se restauraron ${state.importedCount} filas añadidas previamente en este navegador.`; } } catch { restoredMessage = 'No se han aplicado los datos locales guardados porque no son compatibles con esta instantánea.'; }
    refreshPrepared(); bindEvents(); renderForecast(); renderComparison(); renderEvents(); renderData();
    $('#loading-state').classList.add('hidden'); $('#app-content').classList.remove('hidden');
    switchTab(location.hash.slice(1));
    if (restoredMessage) feedback(restoredMessage);
  } catch (error) {
    $('#loading-state').classList.add('hidden'); const node = $('#fatal-error'); node.classList.remove('hidden');
    const heading = document.createElement('h2'); heading.textContent = 'No se pudo preparar la aplicación'; const message = document.createElement('p'); message.textContent = error.message || 'No están disponibles los archivos necesarios.';
    const help = document.createElement('p'); help.textContent = 'Abre la versión empaquetada o inicia el servidor local indicado en el README. No se mostrarán precios ni predicciones de ejemplo.'; node.replaceChildren(heading, message, help);
    console.error(error);
  }
}
start();
