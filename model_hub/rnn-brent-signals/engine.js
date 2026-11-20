/** Causal browser inference for the frozen RNN. No fitting or network requests. */
export const RAW_COLUMNS = Object.freeze([
  'brent_conocido', 'wti_conocido', 'vix_conocido', 'oro_conocido',
  'cobre_conocido', 'indice_dolar_conocido', 'brent_fut_Close_conocido',
]);
export const SIGNAL_COLUMNS = Object.freeze([
  'brent_conocido', 'brent_change_1', 'brent_change_5', 'brent_change_21', 'brent_change_63',
  'brent_volatility_21', 'brent_volatility_63', 'brent_ma_distance_63',
  'brent_ma_distance_252', 'brent_drawdown_252', 'wti_asinh_change',
  'gold_change', 'copper_change', 'usd_change', 'log_vix', 'brent_basis',
]);
const DAY = 86_400_000;
const finite = value => typeof value === 'number' && Number.isFinite(value);
const positive = new Set([0, 2, 3, 4, 5, 6]);

function dateValue(value, weekday = true) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) {
    throw new Error('Las fechas deben tener formato AAAA-MM-DD.');
  }
  const result = new Date(`${value}T00:00:00Z`);
  if (!Number.isFinite(result.getTime()) || result.toISOString().slice(0, 10) !== value) {
    throw new Error(`Fecha no válida: ${value}.`);
  }
  if (weekday && [0, 6].includes(result.getUTCDay())) {
    throw new Error(`La fecha ${value} cae en fin de semana; se requieren días de lunes a viernes.`);
  }
  return result;
}

/** Weekdays, including exchange/publication holidays: the model's original calendar. */
export function addBusinessDays(value, count) {
  const day = dateValue(value, false);
  if (!Number.isSafeInteger(count) || count < 0 || count > 10000) {
    throw new Error('El horizonte debe ser un entero entre 0 y 10000.');
  }
  for (let remaining = count; remaining > 0;) {
    day.setTime(day.getTime() + DAY);
    if (![0, 6].includes(day.getUTCDay())) remaining--;
  }
  return day.toISOString().slice(0, 10);
}

function canonicalRows(data, allowMissing = true) {
  if (!data || !Array.isArray(data.columns) || !Array.isArray(data.rows)) {
    throw new Error('La tabla debe incluir columns y rows.');
  }
  const required = ['Date', ...RAW_COLUMNS];
  if (data.columns.length !== required.length || new Set(data.columns).size !== required.length ||
      required.some(name => !data.columns.includes(name))) {
    throw new Error(`Las columnas requeridas son: ${required.join(', ')}.`);
  }
  const indices = required.map(name => data.columns.indexOf(name));
  return data.rows.map((row, index) => {
    if (!Array.isArray(row) || row.length !== required.length) {
      throw new Error(`La fila ${index + 2} no tiene ${required.length} columnas.`);
    }
    const result = indices.map(i => row[i]);
    dateValue(result[0]);
    result.slice(1).forEach((value, i) => {
      if (allowMissing && (value === null || value === '')) { result[i + 1] = null; return; }
      if (!finite(value)) throw new Error(`Valor no numérico en ${RAW_COLUMNS[i]}, ${result[0]}.`);
      if (positive.has(i) && value <= 0) {
        throw new Error(`${RAW_COLUMNS[i]} debe ser positivo en ${result[0]}.`);
      }
    });
    return result;
  });
}

function rolling(values, end, size, operation) {
  if (end + 1 < size) return NaN;
  const subset = values.slice(end - size + 1, end + 1);
  if (!subset.every(finite)) return NaN;
  if (operation === 'max') return Math.max(...subset);
  const mean = subset.reduce((sum, value) => sum + value, 0) / size;
  if (operation === 'mean') return mean;
  return Math.sqrt(subset.reduce((sum, value) => sum + (value - mean) ** 2, 0) / (size - 1));
}

/** Signals exactly follow study.extended.causal_signals; no target price is used. */
export function prepareSeries(data) {
  const rows = canonicalRows(data);
  const dates = rows.map(row => row[0]);
  dates.forEach((date, index) => {
    if (index && dates[index - 1] >= date) throw new Error('Las fechas deben ser únicas y estar ordenadas.');
    if (index && addBusinessDays(dates[index - 1], 1) !== date) {
      throw new Error(`Falta un día del calendario antes de ${date}; completa las entradas conocidas antes de predecir.`);
    }
  });
  const rawRows = rows.map(row => row.slice(1));
  const brent = rawRows.map(row => row[0] ?? NaN);
  const logBrent = brent.map(value => finite(value) ? Math.log(value) : NaN);
  const differences = (values, lag = 1) => values.map((value, i) =>
    i >= lag && finite(value) && finite(values[i - lag]) ? value - values[i - lag] : NaN);
  const changes = [1, 5, 21, 63].map(lag => differences(logBrent, lag));
  const returns = changes[0];
  const wtiChanges = differences(rawRows.map(row => row[1] === null ? NaN : Math.asinh(row[1] / 20)));
  const otherChanges = [3, 4, 5].map(column => differences(rawRows.map(row =>
    row[column] === null ? NaN : Math.log(row[column]))));
  const signals = rawRows.map((row, i) => {
    const values = [
      brent[i], ...changes.map(series => series[i]),
      rolling(returns, i, 21, 'std'), rolling(returns, i, 63, 'std'),
      brent[i] / rolling(brent, i, 63, 'mean') - 1,
      brent[i] / rolling(brent, i, 252, 'mean') - 1,
      brent[i] / rolling(brent, i, 252, 'max') - 1,
      wtiChanges[i], ...otherChanges.map(series => series[i]),
      row[2] === null ? NaN : Math.log(row[2]),
      row[6] === null ? NaN : row[6] / brent[i] - 1,
    ];
    return values.every(finite) ? values : null;
  });
  return { dates, rawRows, signals, signalColumns: [...SIGNAL_COLUMNS],
    firstUsableIndex: signals.findIndex(row => row !== null) };
}

function validateModel(model) {
  if (!model || model.architecture !== 'RNN' || !Number.isInteger(model.window) || model.window < 1) {
    throw new Error('El archivo no contiene una RNN compatible.');
  }
  const names = model.featureColumns;
  const weights = model.weights;
  if (!Array.isArray(names) || names.length !== SIGNAL_COLUMNS.length ||
      new Set(names).size !== names.length || names.some(name => !SIGNAL_COLUMNS.includes(name))) {
    throw new Error('Las señales del modelo no coinciden con las de la aplicación.');
  }
  const hidden = weights?.input?.length;
  const vector = (value, size) => Array.isArray(value) && value.length === size && value.every(finite);
  const matrix = (value, rows, columns) => Array.isArray(value) && value.length === rows &&
    value.every(row => vector(row, columns));
  if (!Number.isInteger(hidden) || hidden < 1 || hidden > 256 ||
      !matrix(weights.input, hidden, names.length) || !matrix(weights.recurrent, hidden, hidden) ||
      !vector(weights.inputBias, hidden) || !vector(weights.recurrentBias, hidden) ||
      !Array.isArray(model.horizons) || !model.horizons.length ||
      model.horizons.some(h => !Number.isInteger(h) || h < 1 || h > 10000) ||
      !matrix(weights.output, model.horizons.length, hidden) ||
      !vector(weights.outputBias, model.horizons.length) ||
      !vector(model.scaler?.median, names.length) || !vector(model.scaler?.scale, names.length) ||
      model.scaler.scale.some(value => value <= 0) ||
      !vector(model.scaler?.targetScale, model.horizons.length) ||
      model.scaler.targetScale.some(value => value <= 0)) {
    throw new Error('Pesos o escalas del modelo no válidos.');
  }
}

/** Frozen float32 inputs, tanh recurrence, last state, residual absolute price. */
export function predictAt(model, prepared, originDate) {
  validateModel(model);
  const end = prepared.dates.indexOf(originDate);
  if (end < 0) throw new Error('La fecha de origen no está en las observaciones.');
  const start = end - model.window + 1;
  if (start < 0 || prepared.signals.slice(start, end + 1).some(row => row === null)) {
    throw new Error(`Falta historial: se requieren ${model.window} días de señales completas, además del calentamiento de 252 días.`);
  }
  const indices = model.featureColumns.map(name => SIGNAL_COLUMNS.indexOf(name));
  const w = model.weights;
  let state = Array(w.input.length).fill(0);
  for (let time = start; time <= end; time++) {
    const inputs = indices.map((index, j) => Math.fround(
      (prepared.signals[time][index] - model.scaler.median[j]) / model.scaler.scale[j]));
    state = state.map((_, unit) => {
      let value = w.inputBias[unit] + w.recurrentBias[unit];
      for (let j = 0; j < inputs.length; j++) value += w.input[unit][j] * inputs[j];
      for (let j = 0; j < state.length; j++) value += w.recurrent[unit][j] * state[j];
      return Math.fround(Math.tanh(value));
    });
  }
  const base = prepared.rawRows[end][0];
  const predictions = model.horizons.map((horizon, output) => {
    let value = w.outputBias[output];
    for (let j = 0; j < state.length; j++) value += w.output[output][j] * state[j];
    const change = value * model.scaler.targetScale[output];
    const price = base + change;
    if (!finite(price)) throw new Error('La red ha producido un resultado no finito.');
    return { horizon, targetDate: addBusinessDays(originDate, horizon), price, change, changePct: 100 * change / base };
  });
  const daily = prepared.signals[end][SIGNAL_COLUMNS.indexOf('brent_volatility_63')];
  return { origin: originDate, base, futureReference: prepared.rawRows[end][6], predictions,
    volatility: { daily, annualized: daily * Math.sqrt(252) } };
}

/** CSV parser with quoted fields, BOM and either common separator. No evaluation. */
export function parseCSV(text) {
  if (typeof text !== 'string') throw new Error('El CSV debe ser texto.');
  text = text.replace(/^\uFEFF/, '');
  const header = text.split(/\r?\n/, 1)[0];
  const separator = header.includes(';') ? ';' : ',';
  const records = [];
  let row = [], field = '', quoted = false, closed = false;
  for (let i = 0; i <= text.length; i++) {
    const char = i === text.length ? '\n' : text[i];
    if (quoted) {
      if (i === text.length) throw new Error('El CSV contiene comillas sin cerrar.');
      if (char === '"') {
        if (text[i + 1] === '"') { field += '"'; i++; }
        else { quoted = false; closed = true; }
      } else field += char;
      continue;
    }
    if (char === '"') {
      if (field.trim() || closed) throw new Error('Comillas mal colocadas en el CSV.');
      field = ''; quoted = true;
    } else if (char === separator || char === '\n' || char === '\r') {
      row.push(field.trim()); field = ''; closed = false;
      if (char !== separator) {
        if (char === '\r' && text[i + 1] === '\n') i++;
        if (row.some(value => value !== '')) records.push(row);
        row = [];
      }
    } else {
      if (closed && char.trim()) throw new Error('Texto después de un campo entre comillas.');
      field += char;
    }
    if (records.length > 10001) throw new Error('Se permiten como máximo 10000 observaciones por CSV.');
  }
  if (records.length < 2) throw new Error('El CSV debe contener cabecera y al menos una observación.');
  const columns = records.shift();
  const dateIndex = columns.indexOf('Date');
  const rows = records.map((record, index) => {
    if (record.length !== columns.length) throw new Error(`Número de columnas incorrecto en la fila ${index + 2}.`);
    return record.map((value, column) => {
      if (column === dateIndex) return value;
      if (value === '') return null;
      // Decimal commas are accepted only with the semicolon delimiter.
      const normalized = separator === ';' ? value.replace(',', '.') : value;
      if (!/^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$/.test(normalized)) {
        throw new Error(`Número no válido en la fila ${index + 2}.`);
      }
      const number = Number(normalized);
      if (!finite(number)) throw new Error(`Número fuera de rango en la fila ${index + 2}.`);
      return number;
    });
  });
  const canonical = canonicalRows({ columns, rows });
  const dates = canonical.map(row => row[0]);
  if (new Set(dates).size !== dates.length) throw new Error('El CSV contiene fechas duplicadas.');
  canonical.sort((left, right) => left[0].localeCompare(right[0]));
  return { columns: ['Date', ...RAW_COLUMNS], rows: canonical };
}

/** Add new known inputs; actual target observations are never synthesized here. */
export function mergeObservations(baseData, imported) {
  const base = canonicalRows(baseData);
  const updates = canonicalRows(imported);
  if (!base.length) throw new Error('La serie de base está vacía.');
  base.forEach((row, i) => {
    if (i && base[i - 1][0] >= row[0]) throw new Error('La serie de base debe estar ordenada y no tener duplicados.');
  });
  if (updates.length > 10000) throw new Error('Se permiten como máximo 10000 observaciones por CSV.');
  const known = new Map(base.map(row => [row[0], row]));
  const latestOriginal = base.at(-1)[0];
  const sorted = [...updates].sort((a, b) => a[0].localeCompare(b[0]));
  if (new Set(sorted.map(row => row[0])).size !== sorted.length) throw new Error('Hay fechas duplicadas en las nuevas observaciones.');
  let added = 0, unchanged = 0, filled = 0;
  for (const row of sorted) {
    if (row[0] <= latestOriginal) {
      const old = known.get(row[0]);
      if (!old || row.some((value, index) => value !== null && value !== old[index])) {
        throw new Error(`La observación ${row[0]} modifica el histórico; solo se permite añadir datos nuevos.`);
      }
      unchanged++;
      continue;
    }
    let next = addBusinessDays(base.at(-1)[0], 1);
    while (next < row[0]) {
      base.push([next, ...base.at(-1).slice(1)]);
      filled++;
      if (filled > 10000) throw new Error('El salto temporal excede 10000 días laborables.');
      next = addBusinessDays(next, 1);
    }
    const previous = base.at(-1);
    const merged = row.map((value, index) => index && value === null ? previous[index] : value);
    canonicalRows({ columns: ['Date', ...RAW_COLUMNS], rows: [merged] }, false);
    base.push(merged);
    added++;
  }
  return { data: { columns: ['Date', ...RAW_COLUMNS], rows: base }, added, unchanged, filled };
}

export function forecastCSV(result, model = null) {
  const columns = ['origin', 'target_date', 'horizon_business_days', 'forecast_usd_per_barrel',
    'lower_80_usd_per_barrel', 'upper_80_usd_per_barrel', 'change_usd_per_barrel', 'change_pct',
    'known_brent_usd_per_barrel', 'known_future_usd_per_barrel', 'model'];
  const quote = value => `"${String(value ?? '').replaceAll('"', '""')}"`;
  const rows = result.predictions.map(point => {
    const width = model?.intervalHalfWidths?.[point.horizon];
    const hasInterval = finite(width) && width >= 0;
    return [result.origin, point.targetDate, point.horizon, point.price,
      hasInterval ? point.price - width : null, hasInterval ? point.price + width : null,
      point.change, point.changePct, result.base, result.futureReference, model?.id ?? 'RNN-signals-seed42'];
  });
  return [columns.join(','), ...rows.map(row => row.map(quote).join(','))].join('\r\n') + '\r\n';
}

export function computeMetrics(rows) {
  if (!Array.isArray(rows)) throw new Error('Las observaciones deben ser una lista.');
  if (rows.some(row => !['actual', 'prediction', 'naive', 'future_naive'].every(key => finite(row[key])))) {
    throw new Error('Las métricas requieren valores reales y referencias numéricas para todas las observaciones.');
  }
  const n = rows.length;
  if (!n) return { n: 0, mae: null, mse: null, rmse: null, r2: null, relativeRmse: null, relativeRmseVsFuture: null,
    relative_rmse: null, relative_rmse_vs_future: null };
  const mean = rows.reduce((sum, row) => sum + row.actual, 0) / n;
  let absolute = 0, squared = 0, variance = 0, naiveSquared = 0, futureSquared = 0;
  rows.forEach(row => {
    const error = row.prediction - row.actual;
    absolute += Math.abs(error); squared += error ** 2;
    variance += (row.actual - mean) ** 2;
    naiveSquared += (row.naive - row.actual) ** 2;
    futureSquared += (row.future_naive - row.actual) ** 2;
  });
  const relativeRmse = naiveSquared ? Math.sqrt(squared / naiveSquared) : null;
  const relativeRmseVsFuture = futureSquared ? Math.sqrt(squared / futureSquared) : null;
  return { n, mae: absolute / n, mse: squared / n, rmse: Math.sqrt(squared / n),
    r2: variance ? 1 - squared / variance : null,
    relativeRmse, relativeRmseVsFuture, relative_rmse: relativeRmse, relative_rmse_vs_future: relativeRmseVsFuture };
}
