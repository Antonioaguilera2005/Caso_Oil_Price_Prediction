import test from 'node:test';
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';

// Data URL works with engine.js in a plain static HTML project (no package.json needed).
const source = await readFile(new URL('./engine.js', import.meta.url), 'utf8');
const { RAW_COLUMNS, SIGNAL_COLUMNS, prepareSeries, predictAt, addBusinessDays,
  parseCSV, mergeObservations, forecastCSV, computeMetrics } =
  await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);

function fixture(count = 400) {
  let date = '2020-01-01';
  const rows = Array.from({ length: count }, (_, i) => {
    const row = [date, 70 + i / 100 + Math.sin(i / 7), i === 270 ? -37 : 60 + i / 100,
      20 + Math.sin(i), 1600 + i, 4 + i / 1000, 100 + i / 100, 71 + i / 100];
    date = addBusinessDays(date, 1);
    return row;
  });
  return { columns: ['Date', ...RAW_COLUMNS], rows };
}
function zeroModel() {
  return { id: 'test', architecture: 'RNN', window: 60, horizons: [5, 63, 252], featureColumns: SIGNAL_COLUMNS,
    scaler: { median: Array(16).fill(0), scale: Array(16).fill(1), targetScale: [1, 1, 1] },
    weights: { input: [Array(16).fill(0)], recurrent: [[0]], inputBias: [0], recurrentBias: [0],
      output: [[0], [0], [0]], outputBias: [0, 0, 0] } };
}
const header = ['Date', ...RAW_COLUMNS].join(',');

test('calendar uses weekdays, UTC year boundaries and rejects invalid dates', () => {
  assert.equal(addBusinessDays('2024-12-27', 5), '2025-01-03');
  assert.equal(addBusinessDays('2024-02-28', 1), '2024-02-29');
  assert.throws(() => addBusinessDays('2023-02-29', 5));
  assert.throws(() => addBusinessDays('2024-01-01', -1));
});
test('252-day rolling warmup and negative WTI follow causal signal rules', () => {
  const prepared = prepareSeries(fixture());
  assert.equal(prepared.firstUsableIndex, 251);
  assert.equal(prepared.signals[250], null);
  assert.ok(prepared.signals[270].every(Number.isFinite));
  const mean252 = fixture().rows.slice(0, 252).reduce((sum, row) => sum + row[1], 0) / 252;
  assert.ok(Math.abs(prepared.signals[251][8] - (fixture().rows[251][1] / mean252 - 1)) < 1e-12);
});
test('signals and predictions never change after appending future input observations', () => {
  const full = fixture();
  const partial = { ...full, rows: full.rows.slice(0, 330) };
  const before = prepareSeries(partial);
  const after = prepareSeries(full);
  assert.deepEqual(before.signals, after.signals.slice(0, 330));
  const model = zeroModel();
  model.weights.input[0][1] = 0.2;
  model.weights.recurrent[0][0] = 0.3;
  model.weights.output = [[1], [2], [3]];
  assert.deepEqual(predictAt(model, before, before.dates.at(-1)),
    predictAt(model, after, before.dates.at(-1)));
});
test('tanh recurrence uses saved scaling, recurrent state and absolute residual units', () => {
  const prepared = prepareSeries(fixture());
  const model = zeroModel();
  model.window = 2;
  model.scaler.median[0] = 70;
  model.scaler.scale[0] = 10;
  model.scaler.targetScale = [2, 3, 4];
  model.weights.input[0][0] = 0.5;
  model.weights.recurrent[0][0] = 0.25;
  model.weights.inputBias[0] = 0.1;
  model.weights.recurrentBias[0] = -0.05;
  model.weights.output = [[1], [2], [3]];
  model.weights.outputBias = [0.1, 0.2, 0.3];
  const end = prepared.dates.length - 1;
  let state = 0;
  for (const index of [end - 1, end]) {
    const normalized = Math.fround((prepared.rawRows[index][0] - 70) / 10);
    state = Math.fround(Math.tanh(0.05 + 0.5 * normalized + 0.25 * state));
  }
  const result = predictAt(model, prepared, prepared.dates[end]);
  result.predictions.forEach((point, index) => {
    const expected = result.base + ((index + 1) * state + (index + 1) / 10) * (index + 2);
    assert.ok(Math.abs(point.price - expected) < 1e-10);
  });
});
test('zero residual weights exactly reproduce the known Brent baseline', () => {
  const prepared = prepareSeries(fixture());
  const result = predictAt(zeroModel(), prepared, prepared.dates.at(-1));
  result.predictions.forEach(point => {
    assert.equal(point.price, result.base);
    assert.equal(point.changePct, 0);
    assert.equal(point.targetDate, addBusinessDays(result.origin, point.horizon));
  });
  assert.throws(() => predictAt(zeroModel(), prepared, prepared.dates[280]), /Falta historial/);
  assert.equal(forecastCSV(result).split('\r\n').filter(Boolean).length, 4);
});
test('forecast CSV exports displayed interval bounds, blank unavailable ranges and escaped model ID', () => {
  const prepared = prepareSeries(fixture());
  const model = zeroModel();
  model.id = 'RNN "señales", semilla 42';
  model.intervalHalfWidths = { 5: 4.25, 63: 12, 252: 25 };
  const result = predictAt(model, prepared, prepared.dates.at(-1));
  const csv = forecastCSV(result, model);
  assert.ok(csv.includes('lower_80_usd_per_barrel,upper_80_usd_per_barrel'));
  assert.ok(csv.includes('"RNN ""señales"", semilla 42"'));
  const firstRow = csv.split('\r\n')[1];
  const cells = firstRow.match(/"(?:[^"]|"")*"/g).map(cell => cell.slice(1, -1).replaceAll('""', '"'));
  assert.equal(Number(cells[4]), result.base - 4.25);
  assert.equal(Number(cells[5]), result.base + 4.25);
  assert.equal(cells[10], model.id);
  const noInterval = forecastCSV(result).split('\r\n')[1];
  const fallback = noInterval.match(/"(?:[^"]|"")*"/g).map(cell => cell.slice(1, -1));
  assert.equal(fallback[4], '');
  assert.equal(fallback[5], '');
});
test('leading nulls remain unavailable instead of being replaced with future values', () => {
  const data = fixture();
  data.rows.slice(0, 30).forEach(row => { row[1] = null; });
  const prepared = prepareSeries(data);
  assert.equal(prepared.firstUsableIndex, 281);
  assert.equal(prepared.signals[280], null);
});
test('CSV supports BOM, CRLF, quoted decimals and semicolon separators', () => {
  const text = '\uFEFF' + header.replaceAll(',', ';') + '\r\n2024-01-02;"75,5";-37;20;2000;4;100;76\r\n';
  const result = parseCSV(text);
  assert.equal(result.rows[0][1], 75.5);
  assert.equal(result.rows[0][2], -37);
});
test('CSV rejects duplicate columns, duplicate dates, malformed quotes and nonnumeric inputs', () => {
  assert.throws(() => parseCSV(header.replace('wti_conocido', 'brent_conocido') + '\n2024-01-02,75,70,20,2000,4,100,76'));
  assert.throws(() => parseCSV(header + '\n2024-01-02,75,70,20,2000,4,100,76\n2024-01-02,75,70,20,2000,4,100,76'));
  assert.throws(() => parseCSV(header + '\n2024-01-02,"75,70,20,2000,4,100,76'));
  assert.throws(() => parseCSV(header + '\n2024-01-02,NaN,70,20,2000,4,100,76'));
  assert.throws(() => parseCSV(header + '\n2024-01-06,75,70,20,2000,4,100,76'));
  assert.throws(() => parseCSV(header + '\n2024-01-02,0,70,20,2000,4,100,76'));
});
test('CSV imports append inputs, fill missing weekdays and reject historical changes', () => {
  const base = fixture(400);
  const date = addBusinessDays(base.rows.at(-1)[0], 3);
  const incoming = { columns: base.columns, rows: [[date, 85, null, 20, 2200, 4.5, 105, 86]] };
  const merged = mergeObservations(base, incoming);
  assert.equal(merged.added, 1);
  assert.equal(merged.filled, 2);
  assert.equal(merged.data.rows.at(-1)[2], base.rows.at(-1)[2]);
  assert.equal(base.rows.length, 400);
  assert.equal(mergeObservations(base, { columns: base.columns, rows: [base.rows.at(-1)] }).unchanged, 1);
  assert.throws(() => mergeObservations(base, { columns: base.columns, rows: [[base.rows.at(-1)[0], 99, 60, 20, 2000, 4, 100, 76]] }), /histórico/);
});
test('metric arithmetic includes negative R2 and handles zero reference error', () => {
  const result = computeMetrics([
    { actual: 1, prediction: 3, naive: 2, future_naive: 1 },
    { actual: 3, prediction: 5, naive: 4, future_naive: 3 },
  ]);
  assert.deepEqual(result, { n: 2, mae: 2, mse: 4, rmse: 2, r2: -3, relativeRmse: 2, relativeRmseVsFuture: null,
    relative_rmse: 2, relative_rmse_vs_future: null });
  assert.equal(computeMetrics([]).n, 0);
  assert.throws(() => computeMetrics([{ actual: NaN }]));
});
