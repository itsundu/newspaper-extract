/* ============================================================
   TERRALYTIX PROPERTY INTELLIGENCE
   All analytics are computed client-side from the live Supabase
   dataset (~3.3k rows today) — no chart value is hard-coded.
   At materially larger scale (50k+ rows) these aggregations
   should move server-side; see analytics_recommendations.sql.
   ============================================================ */

const SUPABASE_URL = "https://ivxftbgzxxdacizxwnhc.supabase.co";
const SUPABASE_ANON_KEY = "sb_publishable_ZQzA66pSAJvAV07BHaLZDg_OLxHdcnO";
const supabaseClient = supabase.createClient(SUPABASE_URL, SUPABASE_ANON_KEY);

/* ---------------------------------------------------------------
   THEME (same localStorage key as index.html, so the choice carries
   across both pages)
--------------------------------------------------------------- */
const THEME_KEY = "terralytix_theme";
function systemPrefersDark() {
  return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
}
function currentTheme() {
  const stored = localStorage.getItem(THEME_KEY);
  if (stored === "light" || stored === "dark") return stored;
  return systemPrefersDark() ? "dark" : "light";
}
function applyTheme(theme, persist) {
  if (persist) {
    localStorage.setItem(THEME_KEY, theme);
    document.documentElement.setAttribute("data-theme", theme);
  } else if (localStorage.getItem(THEME_KEY)) {
    document.documentElement.setAttribute("data-theme", localStorage.getItem(THEME_KEY));
  } else {
    document.documentElement.removeAttribute("data-theme");
  }
  const icon = document.getElementById("themeToggleIcon");
  const label = document.getElementById("themeToggleLabel");
  icon.textContent = theme === "dark" ? "☀️" : "🌙";
  label.textContent = theme === "dark" ? "Light" : "Dark";
}
applyTheme(currentTheme());
document.getElementById("themeToggle").addEventListener("click", () => {
  const next = currentTheme() === "dark" ? "light" : "dark";
  applyTheme(next, true);
  renderAll();
});
function isDarkTheme() {
  const forced = document.documentElement.getAttribute("data-theme");
  if (forced) return forced === "dark";
  return systemPrefersDark();
}
function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/* ---------------------------------------------------------------
   STAT UTILITIES
--------------------------------------------------------------- */
function num(v) {
  if (v === null || v === undefined || v === "") return null;
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
}
function median(vals) {
  if (!vals || !vals.length) return null;
  const s = [...vals].sort((a, b) => a - b);
  const n = s.length, mid = Math.floor(n / 2);
  return n % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}
function mean(vals) {
  if (!vals || !vals.length) return null;
  return vals.reduce((a, b) => a + b, 0) / vals.length;
}
function percentile(vals, p) {
  if (!vals || !vals.length) return null;
  const s = [...vals].sort((a, b) => a - b);
  const idx = (p / 100) * (s.length - 1);
  const lo = Math.floor(idx), hi = Math.ceil(idx);
  if (lo === hi) return s[lo];
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}
function iqrBounds(vals, k) {
  const q1 = percentile(vals, 25), q3 = percentile(vals, 75);
  if (q1 === null || q3 === null) return { lo: -Infinity, hi: Infinity };
  const iqr = q3 - q1;
  return { lo: q1 - k * iqr, hi: q3 + k * iqr };
}
function linearRegression(points) {
  const n = points.length;
  if (n < 2) return null;
  let sx = 0, sy = 0, sxy = 0, sxx = 0;
  for (const { x, y } of points) { sx += x; sy += y; sxy += x * y; sxx += x * x; }
  const denom = n * sxx - sx * sx;
  if (denom === 0) return null;
  const slope = (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  return { slope, intercept };
}
function groupBy(arr, keyFn) {
  const map = new Map();
  for (const item of arr) {
    const k = keyFn(item);
    if (k === null || k === undefined) continue;
    if (!map.has(k)) map.set(k, []);
    map.get(k).push(item);
  }
  return map;
}

/* ---------------------------------------------------------------
   FORMATTING
--------------------------------------------------------------- */
function fmtINR(v) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  if (v >= 10000000) return "₹" + (v / 10000000).toFixed(v % 10000000 === 0 ? 0 : 2) + " Cr";
  if (v >= 100000) return "₹" + (v / 100000).toFixed(v % 100000 === 0 ? 0 : 1) + " L";
  if (v >= 1000) return "₹" + (v / 1000).toFixed(v % 1000 === 0 ? 0 : 1) + "k";
  return "₹" + Math.round(v);
}
function fmtNum(v) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return Math.round(v).toLocaleString("en-IN");
}
function fmtPct(v, digits) {
  if (v === null || v === undefined || isNaN(v)) return "—";
  return v.toFixed(digits === undefined ? 1 : digits) + "%";
}
function esc(s) {
  return String(s === null || s === undefined ? "" : s).replace(/[&<>"']/g, c => ({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
}

/* ---------------------------------------------------------------
   LOCALITY NORMALIZATION
   Manually curated — spec explicitly warns against blindly merging
   names without validation, so only confirmed spelling/punctuation
   variants of the SAME locality are collapsed here. Distinct but
   nearby areas (e.g. "Mandaveli" vs "Mandavelipakkam") are left
   separate on purpose.
--------------------------------------------------------------- */
const LOCALITY_CANON = {
  "abiramapuram": "Abhiramapuram",
  "r a puram": "R A Puram",
  "r.a.puram": "R A Puram",
  "raja annamalaipuram (ra puram)": "R A Puram",
  "t. nagar": "T Nagar",
  "t nagar": "T Nagar",
};
function canonLocality(raw) {
  if (!raw) return null;
  const trimmed = String(raw).trim().replace(/\s+/g, " ");
  if (!trimmed) return null;
  const key = trimmed.toLowerCase();
  return LOCALITY_CANON[key] || trimmed;
}

/* ---------------------------------------------------------------
   SOURCE-DATE PARSING
   Filenames look like "MTClassifiedsApr112026-1.pdf" — this is a
   genuine signal (the classifieds issue date), not a fabricated
   one. It is NOT a scrape/first-seen timestamp; labeled as such
   everywhere it's shown.
--------------------------------------------------------------- */
const MONTHS = { Jan:0,Feb:1,Mar:2,Apr:3,May:4,Jun:5,Jul:6,Aug:7,Sep:8,Oct:9,Nov:10,Dec:11 };
function parseSourceDate(sourceFile) {
  if (!sourceFile) return null;
  const m = sourceFile.match(/([A-Za-z]{3})(\d{1,2})(\d{4})/);
  if (!m) return null;
  const monthIdx = MONTHS[m[1]];
  if (monthIdx === undefined) return null;
  const d = new Date(Date.UTC(Number(m[3]), monthIdx, Number(m[2])));
  return isNaN(d.getTime()) ? null : d;
}

/* ---------------------------------------------------------------
   ROW NORMALIZATION
   Supabase/PostgREST serializes Postgres `numeric` columns as JSON
   STRINGS (e.g. "35000.0"), not numbers — every numeric field is
   explicitly coerced here. Sane absolute bounds are also applied so
   extraction mis-parses (price=1, rent=8.9 billion, etc.) don't
   poison medians/regressions downstream.
--------------------------------------------------------------- */
function floorBucket(f) {
  if (f === null) return null;
  if (f <= 0) return "Ground";
  if (f <= 3) return "Low-rise (1-3)";
  if (f <= 8) return "Mid-rise (4-8)";
  return "High-rise (9+)";
}
function bhkBucket(b) {
  if (b === null) return null;
  return b >= 5 ? "5+ BHK" : `${b} BHK`;
}
const COMPLETENESS_FIELDS = ["city", "locality", "propertyType", "bhk", "sqft", "uds", "floor", "facing"];
function normalizeRow(r) {
  const price = num(r.price_in_inr);
  const rent = num(r.rent_in_inr);
  const sqft = num(r.sqft_builtup);
  const uds = num(r.sqft_uds);
  const bhk = num(r.bhk);
  const floor = num(r.floor);
  const isRental = !!r.is_rental;

  const row = {
    id: r.ID,
    sourceFile: r.source_file || null,
    sourceDate: parseSourceDate(r.source_file),
    listingText: r.listing_text || null,
    city: (r.city || "").trim() || null,
    locality: canonLocality(r.locality),
    propertyType: (r.property_type || "").trim() || null,
    bhk: bhk !== null ? Math.round(bhk) : null,
    sqft: (sqft !== null && sqft >= 100 && sqft <= 20000) ? sqft : null,
    uds: (uds !== null && uds >= 50 && uds <= 20000) ? uds : null,
    floor: (floor !== null && floor >= 0 && floor <= 60) ? floor : null,
    facing: r.facing || null,
    isRental,
    price: (!isRental && price !== null && price >= 100000 && price <= 1000000000) ? price : null,
    rent: (isRental && rent !== null && rent >= 1000 && rent <= 2000000) ? rent : null,
    contact: r.contact_numbers || null,
  };
  row.ppsf = (row.price !== null && row.sqft !== null) ? row.price / row.sqft : null;
  if (row.ppsf !== null && (row.ppsf < 500 || row.ppsf > 100000)) row.ppsf = null;
  row.floorBucket = floorBucket(row.floor);
  row.bhkBucket = bhkBucket(row.bhk);

  const filled = COMPLETENESS_FIELDS.filter(f => row[f] !== null).length;
  const valueOk = row.isRental ? row.rent !== null : row.price !== null;
  row.completeness = Math.round(((filled + (valueOk ? 1 : 0)) / (COMPLETENESS_FIELDS.length + 1)) * 100);

  return row;
}

/* ---------------------------------------------------------------
   DATA FETCH
--------------------------------------------------------------- */
let ALL = [];      // every normalized row
let FILTERED = []; // rows passing the current global filters

async function fetchAllListings() {
  const FETCH_PAGE = 1000;
  let rows = [];
  let from = 0;
  while (true) {
    const { data, error } = await supabaseClient
      .from("active_listings")
      .select("*")
      .order("ID", { ascending: false })
      .range(from, from + FETCH_PAGE - 1);
    if (error) { console.error("Supabase error:", error); break; }
    if (!data || data.length === 0) break;
    rows = rows.concat(data);
    if (data.length < FETCH_PAGE) break;
    from += FETCH_PAGE;
  }
  return rows;
}

/* ---------------------------------------------------------------
   FILTERS
--------------------------------------------------------------- */
const filters = { city: "", locality: "", propertyType: "", bhk: "", saleRental: "", priceMin: null, priceMax: null, minCohort: 10 };

function populateFilterOptions() {
  const citySel = document.getElementById("fCity");
  const localitySel = document.getElementById("fLocality");
  const typeSel = document.getElementById("fPropertyType");
  const bhkSel = document.getElementById("fBhk");
  const ppsfBhkSel = document.getElementById("ppsfBhkFilter");

  const cities = [...new Set(ALL.map(r => r.city).filter(Boolean))].sort();
  const localities = [...new Set(ALL.map(r => r.locality).filter(Boolean))].sort();
  const types = [...new Set(ALL.map(r => r.propertyType).filter(Boolean))].sort();
  const bhks = [...new Set(ALL.map(r => r.bhk).filter(v => v !== null))].sort((a, b) => a - b);

  const fill = (sel, values, fmt) => {
    const current = sel.value;
    sel.innerHTML = '<option value="">All</option>' + values.map(v => `<option value="${esc(v)}">${esc(fmt ? fmt(v) : v)}</option>`).join("");
    sel.value = current;
  };
  fill(citySel, cities);
  fill(localitySel, localities);
  fill(typeSel, types);
  fill(bhkSel, bhks, v => `${v} BHK`);
  fill(ppsfBhkSel, bhks, v => `${v} BHK`);
}

function applyFilters() {
  filters.city = document.getElementById("fCity").value;
  filters.locality = document.getElementById("fLocality").value;
  filters.propertyType = document.getElementById("fPropertyType").value;
  filters.bhk = document.getElementById("fBhk").value;
  filters.saleRental = document.getElementById("fSaleRental").value;
  filters.priceMin = num(document.getElementById("fPriceMin").value);
  filters.priceMax = num(document.getElementById("fPriceMax").value);
  filters.minCohort = num(document.getElementById("fMinCohort").value) || 10;

  FILTERED = ALL.filter(r => {
    if (filters.city && r.city !== filters.city) return false;
    if (filters.locality && r.locality !== filters.locality) return false;
    if (filters.propertyType && r.propertyType !== filters.propertyType) return false;
    if (filters.bhk && String(r.bhk) !== filters.bhk) return false;
    if (filters.saleRental === "sale" && r.isRental) return false;
    if (filters.saleRental === "rental" && !r.isRental) return false;
    const val = r.isRental ? r.rent : r.price;
    if (filters.priceMin !== null && (val === null || val < filters.priceMin)) return false;
    if (filters.priceMax !== null && (val === null || val > filters.priceMax)) return false;
    return true;
  });

  document.getElementById("filterCount").innerHTML = `<strong>${fmtNum(FILTERED.length)}</strong> of ${fmtNum(ALL.length)} listings match`;
  renderAll();
}

function setFilterAndApply(id, value) {
  document.getElementById(id).value = value;
  applyFilters();
}

document.getElementById("filterBar").addEventListener("change", applyFilters);
document.getElementById("fPriceMin").addEventListener("input", debounce(applyFilters, 400));
document.getElementById("fPriceMax").addEventListener("input", debounce(applyFilters, 400));
document.getElementById("fMinCohort").addEventListener("input", debounce(applyFilters, 400));
document.getElementById("resetFilters").addEventListener("click", () => {
  ["fCity","fLocality","fPropertyType","fBhk","fSaleRental"].forEach(id => document.getElementById(id).value = "");
  document.getElementById("fPriceMin").value = "";
  document.getElementById("fPriceMax").value = "";
  document.getElementById("fMinCohort").value = 10;
  applyFilters();
});
function debounce(fn, ms) {
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}

/* ---------------------------------------------------------------
   CHART REGISTRY + THEME-AWARE OPTIONS
--------------------------------------------------------------- */
const chartRegistry = {};
function makeChart(canvasId, config) {
  if (chartRegistry[canvasId]) chartRegistry[canvasId].destroy();
  const ctx = document.getElementById(canvasId).getContext("2d");
  chartRegistry[canvasId] = new Chart(ctx, config);
  return chartRegistry[canvasId];
}
function baseAxisOptions(xLabel, yLabel) {
  const textColor = cssVar("--chart-text") || "#3b4257";
  const gridColor = cssVar("--chart-grid") || "rgba(15,17,32,0.08)";
  const tickColor = cssVar("--chart-tick") || "#5b6482";
  return {
    responsive: true,
    maintainAspectRatio: false,
    plugins: { legend: { labels: { color: textColor, font: { size: 11 } } } },
    scales: {
      x: { title: xLabel ? { display: true, text: xLabel, color: textColor, font: { size: 11, weight: "600" } } : undefined,
           ticks: { color: tickColor, font: { size: 10 } }, grid: { color: gridColor } },
      y: { title: yLabel ? { display: true, text: yLabel, color: textColor, font: { size: 11, weight: "600" } } : undefined,
           ticks: { color: tickColor, font: { size: 10 } }, grid: { color: gridColor } }
    }
  };
}
const BHK_COLORS = { 1:"#5b8dee", 2:"#4ca22f", 3:"#b7791f", 4:"#c23b3b", 5:"#8b5cf6" };
function bhkColor(bhk) { return BHK_COLORS[bhk] || "#8b93ac"; }

/* ---------------------------------------------------------------
   SECTION: MARKET PULSE
--------------------------------------------------------------- */
function tile(label, value, sub, disabled) {
  return `<div class="pi-stat-tile ${disabled ? "pi-stat-disabled" : ""}">
    <div class="pi-stat-label">${esc(label)}</div>
    <div class="pi-stat-value">${value}</div>
    ${sub ? `<div class="pi-stat-sub">${sub}</div>` : ""}
  </div>`;
}
function renderMarketPulse() {
  const rows = FILTERED;
  const sale = rows.filter(r => !r.isRental);
  const rental = rows.filter(r => r.isRental);
  const cities = new Set(rows.map(r => r.city).filter(Boolean));
  const localities = new Set(rows.map(r => r.locality).filter(Boolean));
  const salePrices = sale.map(r => r.price).filter(v => v !== null);
  const { lo, hi } = iqrBounds(salePrices, 3);
  const cleanPrices = salePrices.filter(v => v >= Math.max(100000, lo) && v <= hi);
  const rents = rental.map(r => r.rent).filter(v => v !== null);
  const { lo: rlo, hi: rhi } = iqrBounds(rents, 3);
  const cleanRents = rents.filter(v => v >= Math.max(1000, rlo) && v <= rhi);
  const ppsfVals = sale.map(r => r.ppsf).filter(v => v !== null);

  const byLocality = groupBy(sale.filter(r => r.ppsf !== null), r => r.locality);
  let mostActive = null, maxCount = 0;
  const localityCounts = groupBy(rows, r => r.locality);
  localityCounts.forEach((v, k) => { if (v.length > maxCount) { maxCount = v.length; mostActive = k; } });

  let highestAvgLocality = null, highestAvg = -Infinity, lowestPpsfLocality = null, lowestPpsf = Infinity;
  byLocality.forEach((v, k) => {
    if (v.length < 5) return;
    const avgP = mean(v.map(r => r.price));
    const medPpsf = median(v.map(r => r.ppsf));
    if (avgP > highestAvg) { highestAvg = avgP; highestAvgLocality = k; }
    if (medPpsf < lowestPpsf) { lowestPpsf = medPpsf; lowestPpsfLocality = k; }
  });

  const tiles = [
    tile("Total Listings", fmtNum(rows.length)),
    tile("Sale Listings", fmtNum(sale.length)),
    tile("Rental Listings", fmtNum(rental.length)),
    tile("Cities Covered", fmtNum(cities.size)),
    tile("Localities Covered", fmtNum(localities.size)),
    tile("Median Price", fmtINR(median(cleanPrices)), "Sale, IQR-trimmed"),
    tile("Average Price", fmtINR(mean(cleanPrices)), "Skewed by large properties"),
    tile("Median ₹/Sqft", fmtNum(median(ppsfVals)), "Across all sale listings"),
    tile("Median Rent", fmtINR(median(cleanRents)), "Per month"),
    tile("Average Rent", fmtINR(mean(cleanRents)), "Per month"),
    tile("Most Active Locality", mostActive ? esc(mostActive) : "—", mostActive ? `${maxCount} listings` : ""),
    tile("Highest Avg. Price Locality", highestAvgLocality ? esc(highestAvgLocality) : "—", highestAvgLocality ? fmtINR(highestAvg) : "", false),
    tile("Lowest ₹/Sqft Locality", lowestPpsfLocality ? esc(lowestPpsfLocality) : "—", lowestPpsfLocality ? `${fmtNum(lowestPpsf)}/sqft` : ""),
    tile("New Listings (7d / 30d)", "Not available", "Requires first_seen_at / scraped_at — see Methodology", true),
  ];
  document.getElementById("pulseGrid").innerHTML = tiles.join("");

  document.getElementById("heroStats").textContent =
    `${fmtNum(ALL.length)}+ listings · ${localities.size || new Set(ALL.map(r=>r.locality).filter(Boolean)).size} localities · ${cities.size || new Set(ALL.map(r=>r.city).filter(Boolean)).size} ${cities.size===1?"city":"cities"} · asking-price data`;

  return { sale, rental, cleanPrices, cleanRents, ppsfVals, mostActive, highestAvgLocality, lowestPpsfLocality };
}

/* ---------------------------------------------------------------
   SECTION: MARKET BRIEF (deterministic NLG — every number here is
   read directly from the stats computed elsewhere on this page)
--------------------------------------------------------------- */
function renderMarketBrief(pulse) {
  const rows = FILTERED;
  if (!rows.length) {
    document.getElementById("marketBrief").textContent = "No listings match the current filters.";
    return;
  }
  const medPrice = median(pulse.cleanPrices);
  const medPpsf = median(pulse.ppsfVals);
  const bhkCounts = groupBy(rows, r => r.bhk);
  let commonBhk = null, commonBhkN = 0;
  bhkCounts.forEach((v, k) => { if (v.length > commonBhkN) { commonBhkN = v.length; commonBhk = k; } });

  const scopeLabel = filters.locality || filters.city || "the current selection";
  const cityMedPpsf = median(ALL.filter(r => !r.isRental && r.ppsf !== null).map(r => r.ppsf));
  let segment = "in line with the city-wide median";
  if (medPpsf && cityMedPpsf) {
    const diff = ((medPpsf - cityMedPpsf) / cityMedPpsf) * 100;
    if (diff > 15) segment = `in the premium segment (+${diff.toFixed(0)}% vs. the city-wide median ₹/sqft)`;
    else if (diff < -15) segment = `in the more affordable segment (${diff.toFixed(0)}% vs. the city-wide median ₹/sqft)`;
  }

  let sentence = `<p><strong>${esc(scopeLabel)}</strong> currently has <strong>${fmtNum(rows.length)}</strong> listings in the dataset` +
    (pulse.sale.length ? ` (${fmtNum(pulse.sale.length)} sale, ${fmtNum(pulse.rental.length)} rental).` : `.`) + `</p>`;
  if (medPrice) sentence += `<p>The median asking price is <strong>${fmtINR(medPrice)}</strong>` +
    (medPpsf ? `, at a median of <strong>₹${fmtNum(medPpsf)}/sqft</strong>` : "") +
    `, placing it ${segment} based on the current dataset.</p>`;
  if (commonBhk !== null) sentence += `<p>The most common configuration here is <strong>${commonBhk} BHK</strong> (${fmtNum(commonBhkN)} of ${fmtNum(rows.length)} listings).</p>`;
  sentence += `<p style="color:var(--muted); font-size:12px;">All figures are asking-price based and computed live from ${fmtNum(rows.length)} currently-filtered listings.</p>`;
  document.getElementById("marketBrief").innerHTML = sentence;
}

/* ---------------------------------------------------------------
   SECTION: PRICE DISTRIBUTION
--------------------------------------------------------------- */
function buildLogHistogram(values, binCount) {
  const positive = values.filter(v => v > 0);
  if (positive.length < 4) return { labels: [], counts: [] };
  const logs = positive.map(v => Math.log10(v));
  const { lo, hi } = iqrBounds(logs, 1.5);
  const trimmed = logs.filter(l => l >= lo && l <= hi);
  if (trimmed.length < 2) return { labels: [], counts: [] };
  const min = Math.min(...trimmed), max = Math.max(...trimmed);
  const binSize = (max - min) / binCount || 1;
  const counts = new Array(binCount).fill(0);
  trimmed.forEach(l => {
    let idx = Math.floor((l - min) / binSize);
    if (idx >= binCount) idx = binCount - 1;
    if (idx < 0) idx = 0;
    counts[idx]++;
  });
  const labels = counts.map((_, i) => `${fmtINR(Math.pow(10, min + i * binSize))}–${fmtINR(Math.pow(10, min + (i + 1) * binSize))}`);
  return { labels, counts };
}
function renderPriceDistribution(pulse) {
  const { labels, counts } = buildLogHistogram(pulse.cleanPrices, 12);
  const opts = baseAxisOptions("Asking Price (log scale)", "Listings");
  opts.plugins.legend.display = false;
  opts.scales.x.ticks.maxRotation = 60;
  opts.scales.x.ticks.minRotation = 45;
  makeChart("priceDistChart", {
    type: "bar",
    data: { labels, datasets: [{ data: counts, backgroundColor: "rgba(76,162,47,0.7)", borderColor: cssVar("--green") || "#4ca22f", borderWidth: 1, borderRadius: 4 }] },
    options: opts
  });
  const p25 = percentile(pulse.cleanPrices, 25), p75 = percentile(pulse.cleanPrices, 75);
  const med = median(pulse.cleanPrices), avg = mean(pulse.cleanPrices);
  const outliers = FILTERED.filter(r => !r.isRental).length - pulse.cleanPrices.length;
  document.getElementById("priceDistInsight").innerHTML = (p25 && p75)
    ? `50% of listed properties fall between <strong>${fmtINR(p25)}</strong> and <strong>${fmtINR(p75)}</strong> (median <strong>${fmtINR(med)}</strong>, mean ${fmtINR(avg)}). ${outliers > 0 ? `${outliers} sale listing(s) excluded as statistical outliers.` : ""}`
    : "Not enough sale listings in this selection to compute a distribution.";
}

/* ---------------------------------------------------------------
   SECTION: PRICE VS SIZE (scatter + trendline + value-opportunity flag)
--------------------------------------------------------------- */
function renderPriceVsSize(pulse) {
  const pts = pulse.sale.filter(r => r.price !== null && r.sqft !== null && r.bhk !== null);
  const { lo, hi } = iqrBounds(pts.map(r => r.price), 3);
  const clean = pts.filter(r => r.price >= Math.max(100000, lo) && r.price <= hi);
  const reg = linearRegression(clean.map(r => ({ x: r.sqft, y: r.price })));

  const byBhk = groupBy(clean, r => r.bhk);
  const datasets = [];
  const opportunities = [];
  byBhk.forEach((rowsForBhk, bhk) => {
    const predictedFor = r => reg ? reg.slope * r.sqft + reg.intercept : null;
    rowsForBhk.forEach(r => {
      const predicted = predictedFor(r);
      if (predicted && r.price < predicted * 0.8) opportunities.push(r);
    });
    datasets.push({
      label: `${bhk} BHK`,
      data: rowsForBhk.map(r => ({ x: r.sqft, y: r.price, _row: r })),
      backgroundColor: bhkColor(bhk) + "b3",
      borderColor: bhkColor(bhk),
      pointRadius: rowsForBhk.map(r => {
        const predicted = predictedFor(r);
        return (predicted && r.price < predicted * 0.8) ? 6 : 3.5;
      }),
      pointStyle: rowsForBhk.map(r => {
        const predicted = predictedFor(r);
        return (predicted && r.price < predicted * 0.8) ? "star" : "circle";
      }),
      borderWidth: 1,
    });
  });

  if (reg) {
    const xs = clean.map(r => r.sqft);
    const minX = Math.min(...xs), maxX = Math.max(...xs);
    datasets.push({
      label: "Trendline",
      type: "line",
      data: [{ x: minX, y: reg.slope * minX + reg.intercept }, { x: maxX, y: reg.slope * maxX + reg.intercept }],
      borderColor: cssVar("--muted") || "#5b6482",
      borderDash: [6, 4],
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
    });
  }

  const opts = baseAxisOptions("Built-up Sqft", "Price");
  opts.plugins.tooltip = {
    callbacks: {
      label(ctx) {
        const row = ctx.raw._row;
        if (!row) return ctx.dataset.label;
        const ppsf = row.ppsf ? `₹${fmtNum(row.ppsf)}/sqft` : "—";
        return [
          `${row.locality || "Unknown locality"} · ${row.bhk} BHK`,
          `${fmtNum(row.sqft)} sqft · ${fmtINR(row.price)} · ${ppsf}`,
          `Floor ${row.floor ?? "—"} · Facing ${row.facing || "—"}`
        ];
      }
    }
  };
  makeChart("priceSizeChart", { type: "scatter", data: { datasets }, options: opts });

  document.getElementById("priceSizeInsight").innerHTML = opportunities.length
    ? `<strong>${opportunities.length}</strong> listing(s) sit 20%+ below the fitted price trend for their size — flagged with a star marker as a <strong>"Potential Value Opportunity"</strong> signal.`
    : `Not enough comparable sale listings to fit a reliable trendline.`;
}

/* ---------------------------------------------------------------
   SECTION: PRICE / SQFT BY LOCALITY
--------------------------------------------------------------- */
let sortState = {};
function sortableTable(tableId, columns, rows, defaultSortCol) {
  if (!sortState[tableId]) sortState[tableId] = { col: defaultSortCol, dir: -1 };
  const state = sortState[tableId];
  const sorted = [...rows].sort((a, b) => {
    const av = a[state.col], bv = b[state.col];
    if (typeof av === "string") return state.dir * av.localeCompare(bv);
    return state.dir * ((av ?? -Infinity) - (bv ?? -Infinity));
  });
  const table = document.getElementById(tableId);
  table.innerHTML =
    `<thead><tr>${columns.map(c => `<th data-col="${c.key}" class="${state.col === c.key ? "sorted" : ""}">${esc(c.label)}</th>`).join("")}</tr></thead>` +
    `<tbody>${sorted.map(row => `<tr>${columns.map(c => `<td class="${c.link ? "pi-link" : ""}" ${c.link ? `data-locality="${esc(row.__locality || row.locality || "")}"` : ""}>${c.render ? c.render(row) : esc(row[c.key])}</td>`).join("")}</tr>`).join("")}</tbody>`;
  table.querySelectorAll("th").forEach(th => {
    th.addEventListener("click", () => {
      const col = th.dataset.col;
      if (state.col === col) state.dir *= -1; else { state.col = col; state.dir = -1; }
      sortableTable(tableId, columns, rows, defaultSortCol);
    });
  });
  table.querySelectorAll("td.pi-link").forEach(td => {
    td.addEventListener("click", () => openLocalityModal(td.dataset.locality));
  });
}

function localityCohorts(rows, bhkFilter) {
  const sale = rows.filter(r => !r.isRental && r.ppsf !== null && r.locality && (!bhkFilter || r.bhk === bhkFilter));
  return groupBy(sale, r => r.locality);
}

function renderPricePerSqft() {
  const bhkFilterVal = document.getElementById("ppsfBhkFilter").value;
  const bhkFilter = bhkFilterVal ? Number(bhkFilterVal) : null;
  const byLoc = localityCohorts(FILTERED, bhkFilter);
  const minN = filters.minCohort;

  const stats = [...byLoc.entries()]
    .filter(([, v]) => v.length >= minN)
    .map(([loc, v]) => ({
      locality: loc,
      median: Math.round(median(v.map(r => r.ppsf))),
      avg: Math.round(mean(v.map(r => r.ppsf))),
      count: v.length,
      lowest: Math.round(Math.min(...v.map(r => r.ppsf))),
      highest: Math.round(Math.max(...v.map(r => r.ppsf))),
    }))
    .sort((a, b) => b.median - a.median);

  const top = stats.slice(0, 15);
  const opts = baseAxisOptions("Median ₹/Sqft", null);
  opts.plugins.legend.display = false;
  opts.indexAxis = "y";
  opts.onClick = (evt, elements) => { if (elements.length) setFilterAndApply("fLocality", top[elements[0].index].locality); };
  makeChart("ppsfChart", {
    type: "bar",
    data: { labels: top.map(s => s.locality), datasets: [{ data: top.map(s => s.median), backgroundColor: cssVar("--blue-light") || "#5b8dee", borderRadius: 4 }] },
    options: opts
  });

  sortableTable("ppsfTable", [
    { key: "locality", label: "Locality", link: true },
    { key: "median", label: "Median ₹/Sqft", render: r => fmtNum(r.median) },
    { key: "avg", label: "Average ₹/Sqft", render: r => fmtNum(r.avg) },
    { key: "count", label: "Listings" },
    { key: "lowest", label: "Lowest", render: r => fmtNum(r.lowest) },
    { key: "highest", label: "Highest", render: r => fmtNum(r.highest) },
  ], stats, "median");
}

/* ---------------------------------------------------------------
   SECTION: LOCALITY RANKING + MARKET DEPTH
--------------------------------------------------------------- */
function renderLocalityTables() {
  const minN = filters.minCohort;
  const byLoc = groupBy(FILTERED, r => r.locality);

  const rankRows = [];
  const depthRows = [];
  byLoc.forEach((rows, loc) => {
    const saleRows = rows.filter(r => !r.isRental);
    const ppsfVals = saleRows.map(r => r.ppsf).filter(v => v !== null);
    const priceVals = saleRows.map(r => r.price).filter(v => v !== null);
    const bhkCounts = groupBy(rows, r => r.bhk);
    let typicalBhk = null, typicalN = 0;
    bhkCounts.forEach((v, k) => { if (v.length > typicalN) { typicalN = v.length; typicalBhk = k; } });

    if (ppsfVals.length >= minN) {
      rankRows.push({ locality: loc, medianPpsf: Math.round(median(ppsfVals)), medianPrice: Math.round(median(priceVals)), count: ppsfVals.length, typicalBhk });
    }
    depthRows.push({
      locality: loc, total: rows.length, sale: saleRows.length, rental: rows.length - saleRows.length,
      bhk2: (bhkCounts.get(2) || []).length, bhk3: (bhkCounts.get(3) || []).length,
      bhk4plus: rows.filter(r => r.bhk >= 4).length,
      medianPrice: priceVals.length ? Math.round(median(priceVals)) : null,
      medianPpsf: ppsfVals.length ? Math.round(median(ppsfVals)) : null,
    });
  });
  rankRows.sort((a, b) => b.medianPpsf - a.medianPpsf);

  sortableTable("localityRankTable", [
    { key: "locality", label: "Locality", link: true },
    { key: "medianPpsf", label: "Median ₹/Sqft", render: r => fmtNum(r.medianPpsf) },
    { key: "medianPrice", label: "Median Price", render: r => fmtINR(r.medianPrice) },
    { key: "count", label: "Listings" },
    { key: "typicalBhk", label: "Typical BHK", render: r => r.typicalBhk !== null ? `${r.typicalBhk} BHK` : "—" },
  ], rankRows, "medianPpsf");

  sortableTable("localityDepthTable", [
    { key: "locality", label: "Locality", link: true },
    { key: "total", label: "Total" },
    { key: "sale", label: "Sale" },
    { key: "rental", label: "Rental" },
    { key: "bhk2", label: "2 BHK" },
    { key: "bhk3", label: "3 BHK" },
    { key: "bhk4plus", label: "4+ BHK" },
    { key: "medianPrice", label: "Median Price", render: r => fmtINR(r.medianPrice) },
    { key: "medianPpsf", label: "Median ₹/Sqft", render: r => fmtNum(r.medianPpsf) },
  ], depthRows, "total");
}

/* ---------------------------------------------------------------
   SECTION: LOCALITY × BHK HEATMAP
--------------------------------------------------------------- */
function renderHeatmap() {
  const metric = document.getElementById("heatmapMetric").value;
  const topLocalities = [...groupBy(FILTERED, r => r.locality).entries()]
    .sort((a, b) => b[1].length - a[1].length).slice(0, 12).map(([loc]) => loc);
  const bhks = [1, 2, 3, 4, 5];

  function cellValue(loc, bhk) {
    const rows = FILTERED.filter(r => r.locality === loc && (bhk === 5 ? r.bhk >= 5 : r.bhk === bhk));
    if (metric === "count") return rows.length || null;
    if (metric === "rent") { const v = rows.filter(r => r.isRental && r.rent !== null).map(r => r.rent); return v.length ? Math.round(median(v)) : null; }
    const sale = rows.filter(r => !r.isRental);
    if (metric === "price") { const v = sale.map(r => r.price).filter(x => x !== null); return v.length ? Math.round(median(v)) : null; }
    const v = sale.map(r => r.ppsf).filter(x => x !== null);
    return v.length ? Math.round(median(v)) : null;
  }

  const grid = topLocalities.map(loc => bhks.map(b => cellValue(loc, b)));
  const flat = grid.flat().filter(v => v !== null);
  const min = Math.min(...flat), max = Math.max(...flat);

  function colorFor(v) {
    if (v === null || max === min) return null;
    const t = (v - min) / (max - min);
    const g1 = [238, 240, 246], g2 = [76, 162, 47];
    const c = g1.map((c0, i) => Math.round(c0 + (g2[i] - c0) * t));
    return `rgb(${c.join(",")})`;
  }
  const fmtCell = v => metric === "count" ? fmtNum(v) : (metric === "rent" || metric === "price") ? fmtINR(v) : fmtNum(v);

  let html = `<thead><tr><th></th>${bhks.map(b => `<th>${b === 5 ? "5+ BHK" : b + " BHK"}</th>`).join("")}</tr></thead><tbody>`;
  topLocalities.forEach((loc, i) => {
    html += `<tr><td class="pi-hm-label pi-link" data-locality="${esc(loc)}">${esc(loc)}</td>`;
    bhks.forEach((b, j) => {
      const v = grid[i][j];
      html += v === null
        ? `<td class="pi-hm-empty">—</td>`
        : `<td style="background:${colorFor(v)};">${fmtCell(v)}</td>`;
    });
    html += `</tr>`;
  });
  html += `</tbody>`;
  const table = document.getElementById("heatmapTable");
  table.innerHTML = html;
  table.querySelectorAll("td.pi-link").forEach(td => td.addEventListener("click", () => openLocalityModal(td.dataset.locality)));
}

/* ---------------------------------------------------------------
   SECTION: BHK ANALYTICS
--------------------------------------------------------------- */
function renderBhkAnalytics() {
  const byBhk = groupBy(FILTERED.filter(r => r.bhk !== null && r.bhk <= 6), r => r.bhk);
  const bhkKeys = [...byBhk.keys()].sort((a, b) => a - b);
  const stats = bhkKeys.map(bhk => {
    const rows = byBhk.get(bhk);
    const sale = rows.filter(r => !r.isRental);
    const rental = rows.filter(r => r.isRental);
    return {
      bhk, count: rows.length,
      medianPrice: median(sale.map(r => r.price).filter(v => v !== null)),
      medianSqft: median(sale.map(r => r.sqft).filter(v => v !== null)),
      medianPpsf: median(sale.map(r => r.ppsf).filter(v => v !== null)),
      medianRent: median(rental.map(r => r.rent).filter(v => v !== null)),
    };
  });

  sortableTable("bhkTable", [
    { key: "bhk", label: "BHK", render: r => `${r.bhk} BHK` },
    { key: "count", label: "Listings" },
    { key: "medianPrice", label: "Median Price", render: r => fmtINR(r.medianPrice) },
    { key: "medianSqft", label: "Median Sqft", render: r => fmtNum(r.medianSqft) },
    { key: "medianPpsf", label: "Median ₹/Sqft", render: r => fmtNum(r.medianPpsf) },
    { key: "medianRent", label: "Median Rent", render: r => fmtINR(r.medianRent) },
  ], stats, "bhk");

  const mkBarOpts = onClick => {
    const o = baseAxisOptions(null, null);
    o.plugins.legend.display = false;
    if (onClick) o.onClick = onClick;
    return o;
  };
  const clickToBhk = (evt, els) => { if (els.length) setFilterAndApply("fBhk", String(stats[els[0].index].bhk)); };

  makeChart("bhkPriceChart", { type: "bar", data: { labels: stats.map(s => `${s.bhk} BHK`), datasets: [{ data: stats.map(s => s.medianPrice), backgroundColor: cssVar("--green") || "#4ca22f", borderRadius: 4 }] }, options: mkBarOpts(clickToBhk) });
  makeChart("bhkPpsfChart", { type: "bar", data: { labels: stats.map(s => `${s.bhk} BHK`), datasets: [{ data: stats.map(s => s.medianPpsf), backgroundColor: cssVar("--blue-light") || "#5b8dee", borderRadius: 4 }] }, options: mkBarOpts(clickToBhk) });
  makeChart("bhkRentChart", { type: "bar", data: { labels: stats.map(s => `${s.bhk} BHK`), datasets: [{ data: stats.map(s => s.medianRent), backgroundColor: cssVar("--amber") || "#b7791f", borderRadius: 4 }] }, options: mkBarOpts(clickToBhk) });
}

/* ---------------------------------------------------------------
   SECTION: PROPERTY TYPE / FACING / FLOOR
--------------------------------------------------------------- */
function renderCategoricalBreakdowns() {
  function categoryStats(keyFn) {
    const grouped = groupBy(FILTERED.filter(r => keyFn(r) !== null), keyFn);
    return [...grouped.entries()].map(([key, rows]) => {
      const sale = rows.filter(r => !r.isRental);
      return {
        key, count: rows.length,
        medianPrice: median(sale.map(r => r.price).filter(v => v !== null)),
        medianPpsf: median(sale.map(r => r.ppsf).filter(v => v !== null)),
        medianSqft: median(sale.map(r => r.sqft).filter(v => v !== null)),
      };
    }).sort((a, b) => b.count - a.count);
  }

  const typeStats = categoryStats(r => r.propertyType);
  const facingStats = categoryStats(r => r.facing);
  const floorStats = categoryStats(r => r.floorBucket);
  const floorOrder = ["Ground", "Low-rise (1-3)", "Mid-rise (4-8)", "High-rise (9+)"];
  floorStats.sort((a, b) => floorOrder.indexOf(a.key) - floorOrder.indexOf(b.key));

  const barOpts = () => { const o = baseAxisOptions(null, null); o.plugins.legend.display = false; return o; };
  makeChart("typeChart", { type: "bar", data: { labels: typeStats.map(s => s.key), datasets: [{ data: typeStats.map(s => s.count), backgroundColor: cssVar("--green") || "#4ca22f", borderRadius: 4 }] }, options: barOpts() });
  makeChart("facingChart", { type: "bar", data: { labels: facingStats.map(s => s.key), datasets: [{ data: facingStats.map(s => s.count), backgroundColor: cssVar("--blue-light") || "#5b8dee", borderRadius: 4 }] }, options: barOpts() });
  makeChart("floorChart", { type: "bar", data: { labels: floorStats.map(s => s.key), datasets: [{ data: floorStats.map(s => s.count), backgroundColor: cssVar("--amber") || "#b7791f", borderRadius: 4 }] }, options: barOpts() });
}

/* ---------------------------------------------------------------
   SECTION: RENTAL INTELLIGENCE + INDICATIVE YIELD
--------------------------------------------------------------- */
function renderRentalIntelligence() {
  const rental = FILTERED.filter(r => r.isRental && r.rent !== null);
  const { lo, hi } = iqrBounds(rental.map(r => r.rent), 3);
  const clean = rental.filter(r => r.rent >= Math.max(1000, lo) && r.rent <= hi);
  const medRent = median(clean.map(r => r.rent));
  const avgRent = mean(clean.map(r => r.rent));
  const rentPpsf = clean.filter(r => r.sqft !== null).map(r => r.rent / r.sqft);

  document.getElementById("rentalPulseGrid").innerHTML = [
    tile("Rental Listings", fmtNum(rental.length)),
    tile("Median Rent", fmtINR(medRent), "Per month"),
    tile("Average Rent", fmtINR(avgRent), "Per month"),
    tile("Median Rent/Sqft", rentPpsf.length ? "₹" + median(rentPpsf).toFixed(1) : "—", "Per month, per sqft"),
  ].join("");

  const byLoc = groupBy(clean.filter(r => r.locality), r => r.locality);
  const locStats = [...byLoc.entries()].filter(([, v]) => v.length >= Math.min(5, filters.minCohort))
    .map(([loc, v]) => ({ loc, median: Math.round(median(v.map(r => r.rent))), n: v.length }))
    .sort((a, b) => b.median - a.median).slice(0, 12);
  const opts1 = baseAxisOptions("Median Rent", null); opts1.plugins.legend.display = false; opts1.indexAxis = "y";
  opts1.onClick = (evt, els) => { if (els.length) setFilterAndApply("fLocality", locStats[els[0].index].loc); };
  makeChart("rentLocalityChart", { type: "bar", data: { labels: locStats.map(s => s.loc), datasets: [{ data: locStats.map(s => s.median), backgroundColor: cssVar("--amber") || "#b7791f", borderRadius: 4 }] }, options: opts1 });

  const opts2 = baseAxisOptions("Sqft", "Rent");
  opts2.plugins.legend.display = false;
  opts2.plugins.tooltip = { callbacks: { label(ctx) { const r = ctx.raw._row; return r ? [`${r.locality || "—"} · ${r.bhk ?? "—"} BHK`, `${fmtNum(r.sqft)} sqft · ${fmtINR(r.rent)}/mo`] : ""; } } };
  makeChart("rentSizeChart", { type: "scatter", data: { datasets: [{ data: clean.filter(r => r.sqft !== null).map(r => ({ x: r.sqft, y: r.rent, _row: r })), backgroundColor: (cssVar("--amber") || "#b7791f") + "b3", borderColor: cssVar("--amber") || "#b7791f" }] }, options: opts2 });

  // Indicative Gross Rental Yield — cohort (locality x BHK) median rent*12 / median sale price
  const minN = Math.min(3, filters.minCohort);
  const cohortKey = r => `${r.locality}||${r.bhk}`;
  const saleCohorts = groupBy(FILTERED.filter(r => !r.isRental && r.price !== null && r.locality && r.bhk !== null), cohortKey);
  const rentCohorts = groupBy(FILTERED.filter(r => r.isRental && r.rent !== null && r.locality && r.bhk !== null), cohortKey);
  const yieldRows = [];
  saleCohorts.forEach((saleRows, key) => {
    const rentRows = rentCohorts.get(key);
    if (!rentRows || rentRows.length < minN || saleRows.length < minN) return;
    const [loc, bhk] = key.split("||");
    const mp = median(saleRows.map(r => r.price));
    const mr = median(rentRows.map(r => r.rent));
    yieldRows.push({ locality: loc, bhk: Number(bhk), medianPrice: mp, medianRent: mr, yieldPct: (mr * 12 / mp) * 100, nSale: saleRows.length, nRent: rentRows.length });
  });
  yieldRows.sort((a, b) => b.yieldPct - a.yieldPct);
  sortableTable("yieldTable", [
    { key: "locality", label: "Locality", link: true },
    { key: "bhk", label: "BHK", render: r => `${r.bhk} BHK` },
    { key: "medianPrice", label: "Median Price", render: r => fmtINR(r.medianPrice) },
    { key: "medianRent", label: "Median Rent", render: r => fmtINR(r.medianRent) },
    { key: "yieldPct", label: "Indicative Gross Yield", render: r => fmtPct(r.yieldPct, 2) },
    { key: "nSale", label: "Sale Comps" },
    { key: "nRent", label: "Rent Comps" },
  ], yieldRows, "yieldPct");
}

/* ---------------------------------------------------------------
   SECTION: OPPORTUNITY SIGNALS + TERRALYTIX OPPORTUNITY SCORE
   Score components (documented in-page too):
     - Benchmark discount/premium magnitude   (0-40 pts)
     - Comparable sample size (confidence)    (0-25 pts)
     - Listing data completeness              (0-20 pts)
     - Indicative yield vs. city median       (0-15 pts, sale only)
   Freshness is intentionally NOT scored — the dataset has no
   first_seen_at/scraped_at yet (see Methodology).
--------------------------------------------------------------- */
function computeOpportunitySignals() {
  const sale = FILTERED.filter(r => !r.isRental && r.price !== null && r.ppsf !== null && r.locality);
  const minN = Math.max(5, Math.min(filters.minCohort, 15));

  const cohortKey = r => `${r.locality}||${r.bhk}`;
  const rentCohorts = groupBy(FILTERED.filter(r => r.isRental && r.rent !== null && r.locality && r.bhk !== null), cohortKey);

  function benchmarkFor(row) {
    const sameLB = sale.filter(r => r.locality === row.locality && r.bhk === row.bhk && r.id !== row.id);
    if (sameLB.length >= minN) return { median: median(sameLB.map(r => r.ppsf)), n: sameLB.length, scope: `${row.locality}, ${row.bhk} BHK` };
    const sameLoc = sale.filter(r => r.locality === row.locality && r.id !== row.id);
    if (sameLoc.length >= minN) return { median: median(sameLoc.map(r => r.ppsf)), n: sameLoc.length, scope: row.locality };
    return null;
  }

  const signals = [];
  for (const row of sale) {
    const bench = benchmarkFor(row);
    if (!bench) continue;
    const pctDiff = ((row.ppsf - bench.median) / bench.median) * 100; // negative = below benchmark

    let yieldPct = null;
    const rc = rentCohorts.get(cohortKey(row));
    if (rc && rc.length >= 3) yieldPct = (median(rc.map(r => r.rent)) * 12 / row.price) * 100;

    const discountMagnitude = Math.max(0, -pctDiff);
    const discountScore = Math.min(40, (discountMagnitude / 30) * 40);
    const confidenceScore = Math.min(25, (bench.n / 30) * 25);
    const completenessScore = (row.completeness / 100) * 20;
    const yieldScore = yieldPct !== null ? Math.min(15, Math.max(0, (yieldPct - 2.5) / 3 * 15)) : 7.5;
    const opportunityScore = Math.round(discountScore + confidenceScore + completenessScore + yieldScore);
    const confidencePct = Math.round(Math.min(100, (bench.n / 25) * 60 + (row.completeness / 100) * 40));

    signals.push({
      row, benchmark: bench.median, benchScope: bench.scope, benchN: bench.n,
      pctDiff, yieldPct, opportunityScore, confidencePct,
      label: pctDiff <= -10 ? "value" : pctDiff >= 15 ? "premium" : "neutral",
    });
  }
  return signals;
}

function signalCard(sig) {
  const r = sig.row;
  const scoreClass = sig.opportunityScore >= 70 ? "" : sig.opportunityScore >= 45 ? "pi-score-mid" : "pi-score-low";
  const badgeClass = sig.label === "value" ? "pi-badge-value" : sig.label === "premium" ? "pi-badge-premium" : "pi-badge-neutral";
  const badgeText = sig.label === "value" ? "Below Locality Benchmark" : sig.label === "premium" ? "Premium Pricing Signal" : "In Line With Benchmark";
  return `<div class="pi-signal-card">
    <div class="pi-signal-top">
      <div>
        <div class="pi-signal-price">${fmtINR(r.price)} · ${r.bhk} BHK</div>
        <div class="pi-signal-meta">${esc(r.locality || "Unknown locality")} · ₹${fmtNum(r.ppsf)}/sqft</div>
      </div>
      <div class="pi-score-ring ${scoreClass}">${sig.opportunityScore}</div>
    </div>
    <span class="pi-badge ${badgeClass}">${badgeText}</span>
    <span class="pi-badge" style="margin-left:6px; background:transparent; color:var(--muted); font-weight:600;">${sig.pctDiff <= 0 ? "" : "+"}${sig.pctDiff.toFixed(1)}% vs. ${esc(sig.benchScope)} benchmark (₹${fmtNum(sig.benchmark)}/sqft, n=${sig.benchN})</span>
    <div class="pi-signal-metrics">
      <div><span>Indicative yield: </span>${sig.yieldPct !== null ? fmtPct(sig.yieldPct, 1) : "n/a"}</div>
      <div><span>Confidence: </span>${sig.confidencePct}%</div>
      <div><span>Sqft: </span>${fmtNum(r.sqft)}</div>
      <div><span>Data completeness: </span>${r.completeness}%</div>
    </div>
  </div>`;
}

function renderOpportunitySignals() {
  const signals = computeOpportunitySignals();
  const value = signals.filter(s => s.label === "value").sort((a, b) => b.opportunityScore - a.opportunityScore).slice(0, 10);
  const premium = signals.filter(s => s.label === "premium").sort((a, b) => b.pctDiff - a.pctDiff).slice(0, 10);

  document.getElementById("valueSignalCards").innerHTML = value.length
    ? value.map(signalCard).join("")
    : `<div class="pi-caveat">No listings meet the comparable-sample threshold for a value signal under the current filters.</div>`;
  document.getElementById("premiumSignalCards").innerHTML = premium.length
    ? premium.map(signalCard).join("")
    : `<div class="pi-caveat">No listings meet the comparable-sample threshold for a premium signal under the current filters.</div>`;

  document.getElementById("scoreMethodology").innerHTML =
    `<strong>Terralytix Opportunity Score (0–100)</strong> blends: benchmark discount/premium magnitude vs. comparable listings (up to 40 pts),
    comparable sample size / confidence (up to 25 pts), listing data completeness (up to 20 pts), and indicative rental yield vs. a neutral baseline (up to 15 pts).
    It is a statistical signal derived entirely from asking-price data in the current dataset — <strong>not an AI valuation, and not a claim that any property is undervalued.</strong>
    Freshness is not yet scored because the dataset has no listing-date field (see Methodology).`;
}

/* ---------------------------------------------------------------
   SECTION: DATA QUALITY + DUPLICATES
--------------------------------------------------------------- */
function renderDataQuality() {
  const rows = FILTERED;
  const scores = rows.map(r => r.completeness);
  const complete = scores.filter(s => s >= 80).length;
  const missing = field => rows.filter(r => r[field] === null).length;

  document.getElementById("qualityGrid").innerHTML = [
    tile("Avg. Completeness", fmtPct(mean(scores), 0)),
    tile("Complete Listings (≥80%)", fmtNum(complete)),
    tile("Partial Listings (<80%)", fmtNum(rows.length - complete)),
    tile("Missing Price/Rent", fmtNum(rows.filter(r => (r.isRental ? r.rent : r.price) === null).length)),
    tile("Missing Sqft", fmtNum(missing("sqft"))),
    tile("Missing Locality", fmtNum(missing("locality"))),
    tile("Missing BHK", fmtNum(missing("bhk"))),
    tile("Missing Facing", fmtNum(missing("facing"))),
  ].join("");

  const buckets = [0, 0, 0, 0, 0]; // <20 20-40 40-60 60-80 80-100
  scores.forEach(s => { const i = Math.min(4, Math.floor(s / 20)); buckets[i]++; });
  const opts = baseAxisOptions(null, "Listings"); opts.plugins.legend.display = false;
  makeChart("completenessChart", { type: "bar", data: { labels: ["0-20%","20-40%","40-60%","60-80%","80-100%"], datasets: [{ data: buckets, backgroundColor: cssVar("--green") || "#4ca22f", borderRadius: 4 }] }, options: opts });

  // Duplicate heuristic: same contact number, group by similarity
  const byContact = groupBy(rows.filter(r => r.contact), r => r.contact);
  const groups = [];
  byContact.forEach((v, contact) => {
    if (v.length < 2) return;
    const localities = new Set(v.map(r => r.locality));
    const bhks = new Set(v.map(r => r.bhk));
    let confidence = "Low";
    if (localities.size === 1 && bhks.size === 1) confidence = "High";
    else if (localities.size <= 2) confidence = "Medium";
    const localityList = [...localities].filter(Boolean);
    const localitiesLabel = localityList.length > 3
      ? `${localityList.slice(0, 3).join(", ")} +${localityList.length - 3} more`
      : (localityList.join(", ") || "—");
    groups.push({ contact, count: v.length, localities: localitiesLabel, confidence });
  });
  groups.sort((a, b) => b.count - a.count);
  sortableTable("duplicatesTable", [
    { key: "contact", label: "Contact" },
    { key: "count", label: "Listings" },
    { key: "localities", label: "Localities" },
    { key: "confidence", label: "Confidence", render: r => `<span class="pi-badge ${r.confidence === "High" ? "pi-badge-high" : r.confidence === "Medium" ? "pi-badge-medium" : "pi-badge-low"}">${r.confidence}</span>` },
  ], groups.slice(0, 100), "count");

  // Collapsed by default — only show the expand toggle if there's enough
  // content to actually need collapsing.
  const dupWrap = document.getElementById("duplicatesWrap");
  const dupToggle = document.getElementById("duplicatesToggle");
  dupWrap.classList.remove("pi-expanded");
  const shownGroups = Math.min(groups.length, 100);
  dupToggle.dataset.total = shownGroups;
  dupToggle.style.display = shownGroups > 6 ? "" : "none";
  dupToggle.textContent = `Show all ${shownGroups} groups`;
}

/* ---------------------------------------------------------------
   SECTION: SOURCE INTELLIGENCE + CITY INTELLIGENCE
--------------------------------------------------------------- */
function renderSourceAndCity() {
  const bySource = groupBy(FILTERED.filter(r => r.sourceFile), r => r.sourceFile);
  const sourceStats = [...bySource.entries()].map(([src, v]) => ({ src, count: v.length })).sort((a, b) => b.count - a.count).slice(0, 15);
  const opts1 = baseAxisOptions(null, "Listings"); opts1.plugins.legend.display = false;
  opts1.scales.x.ticks.maxRotation = 70; opts1.scales.x.ticks.minRotation = 50; opts1.scales.x.ticks.font = { size: 8 };
  makeChart("sourceChart", { type: "bar", data: { labels: sourceStats.map(s => s.src.replace(/\.pdf$/i, "")), datasets: [{ data: sourceStats.map(s => s.count), backgroundColor: cssVar("--navy") || "#10284f", borderRadius: 4 }] }, options: opts1 });

  const byWeek = groupBy(FILTERED.filter(r => r.sourceDate), r => {
    const d = r.sourceDate;
    const onejan = new Date(Date.UTC(d.getUTCFullYear(), 0, 1));
    const week = Math.ceil((((d - onejan) / 86400000) + onejan.getUTCDay() + 1) / 7);
    return `${d.getUTCFullYear()}-W${String(week).padStart(2, "0")}`;
  });
  const weekKeys = [...byWeek.keys()].sort();
  const opts2 = baseAxisOptions(null, "Listings"); opts2.plugins.legend.display = false;
  opts2.scales.x.ticks.maxRotation = 60; opts2.scales.x.ticks.font = { size: 9 };
  makeChart("timeChart", { type: "line", data: { labels: weekKeys, datasets: [{ data: weekKeys.map(k => byWeek.get(k).length), borderColor: cssVar("--green") || "#4ca22f", backgroundColor: "rgba(76,162,47,0.15)", fill: true, tension: 0.25, pointRadius: 2 }] }, options: opts2 });

  const byCity = groupBy(FILTERED, r => r.city);
  const cityRows = [...byCity.entries()].map(([city, rows]) => {
    const sale = rows.filter(r => !r.isRental);
    const rental = rows.filter(r => r.isRental);
    const byLoc = groupBy(rows, r => r.locality);
    let topLoc = null, topN = 0;
    byLoc.forEach((v, k) => { if (v.length > topN) { topN = v.length; topLoc = k; } });
    const bhkCounts = groupBy(rows, r => r.bhk);
    let commonBhk = null, commonN = 0;
    bhkCounts.forEach((v, k) => { if (v.length > commonN) { commonN = v.length; commonBhk = k; } });
    const medPrice = median(sale.map(r => r.price).filter(v => v !== null));
    const medRent = median(rental.map(r => r.rent).filter(v => v !== null));
    return {
      city, listings: rows.length,
      medianPrice: medPrice, medianPpsf: median(sale.map(r => r.ppsf).filter(v => v !== null)),
      medianRent: medRent, indicativeYield: (medPrice && medRent) ? (medRent * 12 / medPrice) * 100 : null,
      topLocality: topLoc, commonBhk,
    };
  }).sort((a, b) => b.listings - a.listings);

  sortableTable("cityTable", [
    { key: "city", label: "City" },
    { key: "listings", label: "Listings" },
    { key: "medianPrice", label: "Median Price", render: r => fmtINR(r.medianPrice) },
    { key: "medianPpsf", label: "Median ₹/Sqft", render: r => fmtNum(r.medianPpsf) },
    { key: "medianRent", label: "Median Rent", render: r => fmtINR(r.medianRent) },
    { key: "indicativeYield", label: "Indicative Yield", render: r => fmtPct(r.indicativeYield, 2) },
    { key: "topLocality", label: "Top Locality" },
    { key: "commonBhk", label: "Most Common BHK", render: r => r.commonBhk !== null ? `${r.commonBhk} BHK` : "—" },
  ], cityRows, "listings");
}

/* ---------------------------------------------------------------
   METHODOLOGY + ROADMAP (static content)
--------------------------------------------------------------- */
function renderMethodology() {
  document.getElementById("methodologyList").innerHTML = [
    "All prices and rents shown are <strong>asking prices from listings</strong>, not verified transaction data.",
    "The underlying data is extracted from newspaper/classifieds sources and may contain duplicates, typos, or incomplete records — see Data Quality.",
    "Medians are used instead of means wherever a metric is meaningfully skewed (real-estate prices heavily are).",
    "Extreme values are excluded from statistics via IQR-based outlier trimming, not deleted from the underlying dataset.",
    "Locality names are canonicalized only for confirmed spelling/punctuation duplicates (e.g. \"R.A.Puram\" → \"R A Puram\") — distinct nearby areas are never auto-merged.",
    "The Market Brief and Opportunity Score are computed directly from dataset statistics — no language model is involved, and no figure is fabricated.",
    "\"Indicative Gross Rental Yield\" is a locality+BHK cohort estimate (median rent × 12 ÷ median price), not a per-property valuation, and excludes maintenance, taxes and vacancy.",
  ].map(li => `<li>${li}</li>`).join("");

  const roadmap = [
    ["High", "Listing timestamps (first_seen_at, last_seen_at, scraped_at)", "Unlocks true trend analytics: price movement over time, inventory changes, MoM/YoY — see analytics_recommendations.sql."],
    ["Medium", "Server-side aggregation", "Move locality/BHK/city rollups into Postgres views or RPC functions once the dataset materially exceeds ~20-50k rows, so the browser stops fetching the full table."],
    ["Medium", "Map Intelligence", "Requires reliable per-listing geocoding; not fabricated here since it isn't in the current dataset."],
    ["Low", "Ask Terralytix (natural-language query)", "Needs a backend LLM integration — out of scope for a static, client-only page; the anon Supabase key can't safely front an LLM call."],
    ["Low", "Predictive / automated market reports", "Depends on the timestamp history above accumulating over several months."],
  ];
  document.getElementById("roadmapList").innerHTML = roadmap.map(([priority, title, desc]) =>
    `<div class="pi-roadmap-item"><span class="pi-badge ${priority === "High" ? "pi-badge-value" : priority === "Medium" ? "pi-badge-medium" : "pi-badge-low"}">${priority}</span><div><strong>${esc(title)}</strong><div style="color:var(--muted); margin-top:2px;">${esc(desc)}</div></div></div>`
  ).join("");
}

/* ---------------------------------------------------------------
   LOCALITY DEEP-DIVE MODAL
--------------------------------------------------------------- */
function openLocalityModal(locality) {
  if (!locality) return;
  const rows = ALL.filter(r => r.locality === locality);
  const sale = rows.filter(r => !r.isRental);
  const rental = rows.filter(r => r.isRental);
  const ppsfVals = sale.map(r => r.ppsf).filter(v => v !== null);
  const priceVals = sale.map(r => r.price).filter(v => v !== null);
  const rentVals = rental.map(r => r.rent).filter(v => v !== null);
  const bhkCounts = groupBy(rows, r => r.bhk);
  let typicalBhk = null, typicalN = 0;
  bhkCounts.forEach((v, k) => { if (v.length > typicalN) { typicalN = v.length; typicalBhk = k; } });
  const medPrice = median(priceVals), medRent = median(rentVals);
  const indicativeYield = (medPrice && medRent) ? (medRent * 12 / medPrice) * 100 : null;

  const city = rows[0] ? rows[0].city : null;
  const peers = [...groupBy(ALL.filter(r => r.city === city && r.locality && r.locality !== locality), r => r.locality).entries()]
    .map(([loc, v]) => ({ loc, n: v.length, medPpsf: median(v.filter(r => !r.isRental && r.ppsf !== null).map(r => r.ppsf)) }))
    .filter(p => p.n >= 5 && p.medPpsf)
    .sort((a, b) => b.n - a.n).slice(0, 4);

  document.getElementById("localityModalTitle").textContent = locality;
  document.getElementById("localityModalBody").innerHTML = `
    <div class="pi-grid pi-grid-4" style="margin-bottom:16px;">
      ${tile("Listings", fmtNum(rows.length))}
      ${tile("Median Price", fmtINR(medPrice))}
      ${tile("Median ₹/Sqft", fmtNum(median(ppsfVals)))}
      ${tile("Median Rent", fmtINR(medRent))}
    </div>
    <div class="pi-grid pi-grid-4" style="margin-bottom:16px;">
      ${tile("Typical BHK", typicalBhk !== null ? `${typicalBhk} BHK` : "—")}
      ${tile("Sale / Rental Split", `${sale.length} / ${rental.length}`)}
      ${tile("Indicative Gross Yield", indicativeYield !== null ? fmtPct(indicativeYield, 2) : "n/a")}
      ${tile("Price Range (P25–P75)", `${fmtINR(percentile(priceVals, 25))} – ${fmtINR(percentile(priceVals, 75))}`)}
    </div>
    <h3 style="font-size:13px; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted); margin:16px 0 8px;">How does ${esc(locality)} compare?</h3>
    <div class="pi-table-wrap"><table class="pi-table"><thead><tr><th>Locality</th><th>Listings</th><th>Median ₹/Sqft</th></tr></thead><tbody>
      <tr style="font-weight:700;"><td>${esc(locality)} (this locality)</td><td>${fmtNum(rows.length)}</td><td>${fmtNum(median(ppsfVals))}</td></tr>
      ${peers.map(p => `<tr><td class="pi-link" data-locality="${esc(p.loc)}">${esc(p.loc)}</td><td>${fmtNum(p.n)}</td><td>${fmtNum(p.medPpsf)}</td></tr>`).join("")}
    </tbody></table></div>
    <div style="margin-top:16px; text-align:right;">
      <button class="pi-filter-reset" id="modalFilterToLocality" type="button" style="margin-left:0;">Filter dashboard to ${esc(locality)}</button>
    </div>
  `;
  document.getElementById("localityModalBody").querySelectorAll("td.pi-link").forEach(td => td.addEventListener("click", () => openLocalityModal(td.dataset.locality)));
  document.getElementById("modalFilterToLocality").addEventListener("click", () => { closeLocalityModal(); setFilterAndApply("fLocality", locality); });
  document.getElementById("localityModalBackdrop").classList.add("pi-open");
}
function closeLocalityModal() { document.getElementById("localityModalBackdrop").classList.remove("pi-open"); }
document.getElementById("localityModalClose").addEventListener("click", closeLocalityModal);
document.getElementById("localityModalBackdrop").addEventListener("click", e => { if (e.target.id === "localityModalBackdrop") closeLocalityModal(); });
document.addEventListener("keydown", e => { if (e.key === "Escape") closeLocalityModal(); });

/* ---------------------------------------------------------------
   SECTION NAV — active link highlighting on scroll
--------------------------------------------------------------- */
function wireSectionNav() {
  const links = [...document.querySelectorAll("#sectionNav a")];
  const sections = links.map(a => document.querySelector(a.getAttribute("href")));
  const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        const id = "#" + entry.target.id;
        links.forEach(a => a.classList.toggle("active", a.getAttribute("href") === id));
      }
    });
  }, { rootMargin: "-30% 0px -60% 0px" });
  sections.forEach(s => s && observer.observe(s));
}

/* ---------------------------------------------------------------
   ORCHESTRATION
--------------------------------------------------------------- */
function renderAll() {
  if (!FILTERED.length && !ALL.length) return;
  const pulse = renderMarketPulse();
  renderMarketBrief(pulse);
  renderPriceDistribution(pulse);
  renderPriceVsSize(pulse);
  renderPricePerSqft();
  renderLocalityTables();
  renderHeatmap();
  renderBhkAnalytics();
  renderCategoricalBreakdowns();
  renderRentalIntelligence();
  renderOpportunitySignals();
  renderDataQuality();
  renderSourceAndCity();
}

document.getElementById("ppsfBhkFilter").addEventListener("change", renderPricePerSqft);
document.getElementById("heatmapMetric").addEventListener("change", renderHeatmap);
document.getElementById("duplicatesToggle").addEventListener("click", () => {
  const wrap = document.getElementById("duplicatesWrap");
  const btn = document.getElementById("duplicatesToggle");
  const expanded = wrap.classList.toggle("pi-expanded");
  btn.textContent = expanded ? "Show less" : `Show all ${btn.dataset.total} groups`;
});

async function refreshFromDatabase() {
  const rawRows = await fetchAllListings();
  ALL = rawRows.map(normalizeRow);
  populateFilterOptions();
  applyFilters();
}

async function init() {
  renderMethodology();
  const raw = await fetchAllListings();
  ALL = raw.map(normalizeRow);
  FILTERED = ALL;
  populateFilterOptions();
  document.getElementById("filterCount").innerHTML = `<strong>${fmtNum(ALL.length)}</strong> of ${fmtNum(ALL.length)} listings match`;
  renderAll();
  wireSectionNav();

  // Realtime push (instant) when Supabase Realtime is enabled for this
  // table, PLUS a periodic poll as a resilient fallback if it isn't —
  // same dual approach as index.html.
  supabaseClient
    .channel("pi_active_listings_changes")
    .on("postgres_changes", { event: "*", schema: "public", table: "active_listings" }, refreshFromDatabase)
    .subscribe();
  setInterval(refreshFromDatabase, 30000);
}
init();
