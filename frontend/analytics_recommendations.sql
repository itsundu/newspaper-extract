-- ============================================================
-- TERRALYTIX PROPERTY INTELLIGENCE — recommended schema changes
-- ============================================================
-- These are RECOMMENDATIONS, not applied automatically. Run them
-- manually in the Supabase SQL editor when ready. Nothing in
-- property-intelligence.js depends on these existing yet — the
-- dashboard degrades gracefully today by labeling time-series
-- metrics "Not available" rather than fabricating them.
--
-- Current known schema (from a live sample row, Aug 2026):
--   active_listings(
--     "ID" bigint, source_file text, listing_text text, city text,
--     locality text, property_type text, bhk numeric, sqft_builtup numeric,
--     sqft_uds numeric, floor numeric, facing text,
--     price_value numeric, price_unit text, price_in_inr numeric,
--     is_rental boolean, rent_value numeric, rent_unit text,
--     rent_in_inr numeric, contact_numbers text
--   )
-- No timestamp column currently exists.


-- ---------------------------------------------------------------
-- 1. TIMESTAMPS (the single highest-value change — see section 28
--    of the product brief). Without these, no trend/MoM/YoY metric
--    can be computed honestly, no matter how the dashboard is built.
-- ---------------------------------------------------------------
alter table active_listings
  add column if not exists first_seen_at timestamptz default now(),
  add column if not exists last_seen_at  timestamptz default now(),
  add column if not exists scraped_at    timestamptz default now();

-- Populate first_seen_at retroactively as best-effort from the
-- classifieds issue date already embedded in source_file where
-- parseable (e.g. "MTClassifiedsApr112026-1.pdf" -> 2026-04-11).
-- This is NOT a substitute for real scrape timestamps going
-- forward, but it back-fills a reasonable value for existing rows
-- instead of leaving them all defaulted to `now()`.
--
-- update active_listings
-- set first_seen_at = to_timestamp(
--       substring(source_file from '([A-Za-z]{3}\d{1,2}\d{4})'),
--       'MonDDYYYY'
--     )
-- where source_file ~ '[A-Za-z]{3}\d{1,2}\d{4}'
--   and first_seen_at is null;
--
-- (left commented out — review the parsed dates on a sample before
-- running this against the full table.)

create index if not exists idx_active_listings_first_seen_at on active_listings (first_seen_at);


-- ---------------------------------------------------------------
-- 2. INDEXES for the filters/aggregations the dashboard already runs
--    (cheap now at ~3.3k rows; becomes necessary once this moves
--    server-side at higher row counts).
-- ---------------------------------------------------------------
create index if not exists idx_active_listings_locality       on active_listings (locality);
create index if not exists idx_active_listings_city            on active_listings (city);
create index if not exists idx_active_listings_bhk             on active_listings (bhk);
create index if not exists idx_active_listings_is_rental       on active_listings (is_rental);
create index if not exists idx_active_listings_contact_numbers on active_listings (contact_numbers);


-- ---------------------------------------------------------------
-- 3. SERVER-SIDE AGGREGATION (recommended once the dataset grows
--    materially past ~20-50k rows, so the browser stops fetching
--    every row on every page load). These views mirror the exact
--    client-side logic in property-intelligence.js — locality
--    canonicalization and outlier bounds are NOT reproduced here on
--    purpose, since duplicating that judgment in two places invites
--    drift. Prefer exposing these as `security definer` RPC
--    functions via PostgREST rather than raw views, so the API
--    surface stays stable as the underlying tables evolve.
-- ---------------------------------------------------------------

-- Locality-level rollup (sale side), matching the Price/Sqft-by-
-- Locality and Locality Ranking sections.
create or replace view v_locality_sale_stats as
select
  locality,
  count(*) filter (where is_rental = false and price_in_inr is not null) as sale_count,
  percentile_cont(0.5) within group (order by price_in_inr)
    filter (where is_rental = false) as median_price,
  percentile_cont(0.5) within group (order by (price_in_inr / nullif(sqft_builtup, 0)))
    filter (where is_rental = false and sqft_builtup > 0) as median_ppsf
from active_listings
where locality is not null
group by locality;

-- BHK-level rollup, matching the BHK Analytics section.
create or replace view v_bhk_stats as
select
  round(bhk) as bhk,
  count(*) as listing_count,
  percentile_cont(0.5) within group (order by price_in_inr)
    filter (where is_rental = false) as median_price,
  percentile_cont(0.5) within group (order by rent_in_inr)
    filter (where is_rental = true) as median_rent
from active_listings
where bhk is not null
group by round(bhk);

-- Example RPC wrapper (exposed at /rest/v1/rpc/analytics_locality_stats):
-- create or replace function analytics_locality_stats(min_listings int default 10)
-- returns setof v_locality_sale_stats
-- language sql stable
-- as $$
--   select * from v_locality_sale_stats where sale_count >= min_listings;
-- $$;


-- ---------------------------------------------------------------
-- 4. SUGGESTED FUTURE API SHAPE (once server-side aggregation lands)
--    /api/analytics/market        -> market pulse totals
--    /api/analytics/localities    -> v_locality_sale_stats, paginated
--    /api/analytics/bhk           -> v_bhk_stats
--    /api/analytics/rent          -> rent-side equivalents
--    /api/analytics/opportunities -> precomputed opportunity scores
--    (Today, property-intelligence.js computes all of the above
--    client-side against the full table via PostgREST — fine at the
--    current ~3.3k-row scale, not fine at 100k+.)
-- ---------------------------------------------------------------
