-- ============================================================
-- SIH26166 - Supabase schema. Run this ONCE in Supabase:
-- Dashboard -> SQL Editor -> New query -> paste -> RUN
-- ============================================================

-- Scenes: one row per source product (real OHRC + synthetic stand-in)
create table if not exists scenes (
  id uuid primary key default gen_random_uuid(),
  product_id text unique,                    -- e.g. ch2_ohr_ncp_20210401T2357376656
  source_scene text,                         -- our demo id (ohrc_20210401 / tycho)
  instrument text,
  footprint jsonb,                           -- corner lat/lon from PDS4 label
  metadata jsonb,                            -- full parsed label + sun angles
  created_at timestamptz default now()
);

-- Jobs: one row per pipeline run (stage tracking)
create table if not exists jobs (
  id uuid primary key default gen_random_uuid(),
  stage text default 'match',                -- ingest|geometry|match|verify|evaluate|visualize
  status text default 'pending',             -- pending|running|done|failed
  source_scene text,
  created_at timestamptz default now(),
  updated_at timestamptz default now()
);

-- Matches: keypoint correspondences + verified homography per run
create table if not exists matches (
  id uuid primary key default gen_random_uuid(),
  source_scene text,
  keypoints_source jsonb,                    -- [[x,y], ...] (capped at 500)
  keypoints_ref jsonb,
  homography jsonb,                          -- 3x3 matrix
  match_percentage numeric,
  created_at timestamptz default now()
);

-- Metrics: the numbers we report - NEVER synthetic, never Gemini-generated
create table if not exists metrics (
  id uuid primary key default gen_random_uuid(),
  source_scene text,
  rmse numeric,
  inlier_count int,
  inlier_ratio numeric,
  match_percentage numeric,
  method text,                               -- 'sift' | 'sift+learned'
  ransac_k_derived int,                      -- derived k, not a magic constant
  created_at timestamptz default now()
);

create index if not exists idx_matches_scene on matches(source_scene);
create index if not exists idx_metrics_scene on metrics(source_scene);
create index if not exists idx_jobs_scene on jobs(source_scene);

-- Row Level Security: service role key bypasses RLS, anon key can read.
alter table scenes  enable row level security;
alter table jobs    enable row level security;
alter table matches enable row level security;
alter table metrics enable row level security;

create policy "public read scenes"  on scenes  for select using (true);
create policy "public read metrics" on metrics for select using (true);
-- jobs / matches: no public policy -> writes only via service role (backend)

-- Storage buckets (private) - usually created in the dashboard, but SQL works:
insert into storage.buckets (id, name, public)
values ('raw-tiles','raw-tiles',false),
       ('dem-patches','dem-patches',false),
       ('model-weights','model-weights',false)
on conflict (id) do nothing;
