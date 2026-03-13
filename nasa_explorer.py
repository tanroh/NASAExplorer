"""
NASA Astrophysics Data Explorer
Browses JWST and TESS data via the MAST archive (astroquery)
"""

import streamlit as st
import pandas as pd
from astroquery.mast import Observations
from astropy.coordinates import SkyCoord
import astropy.units as u
from PIL import Image
import requests
from io import BytesIO
import lightkurve as lk
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NASA Astrophysics Explorer",
    page_icon="🔭",
    layout="wide",
)

st.title("🔭 NASA Astrophysics Explorer")
st.caption("Browse JWST and TESS observations via the MAST archive")

# ── Session state — lets featured target buttons pre-fill the search box ─────
if "prefill_target" not in st.session_state:
    st.session_state.prefill_target = "NGC 628"

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Search")

    target_input = st.text_input(
        "Object name or RA Dec",
        value=st.session_state.prefill_target,
        help="Examples:  NGC 628  |  M31  |  261.7 -73.5  |  Trappist-1",
    )

    search_mode = st.radio(
        "Resolve target as",
        ["Object name", "RA / Dec (degrees)"],
        index=0,
    )

    missions = st.multiselect(
        "Missions",
        ["JWST", "TESS"],
        default=["JWST", "TESS"],
    )

    radius_arcmin = st.slider("Search radius (arcmin)", 1, 60, 3)

    max_results = st.slider("Max results per mission", 10, 200, 50)

    run_search = st.button("🔍 Search", use_container_width=True)

# ── Helpers ───────────────────────────────────────────────────────────────────

import time as _time

def _with_retry(fn, retries=3, delay=2):
    """Call fn(), retrying on ConnectionError up to `retries` times."""
    for attempt in range(retries):
        try:
            return fn()
        except (ConnectionError, OSError) as e:
            if attempt < retries - 1:
                _time.sleep(delay * (attempt + 1))
            else:
                raise e

# Solar system bodies astropy can resolve via ephemeris
SOLAR_SYSTEM_BODIES = {
    "mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune",
    "pluto", "moon", "sun", "io", "europa", "ganymede", "callisto",
    "titan", "enceladus", "triton", "ceres", "eris",
}

# Curated fallback list used when live fetch fails or is slow
FALLBACK_TARGETS = [
    "WASP-76", "TRAPPIST-1", "HD 209458", "GJ 1214", "TOI-700",
    "NGC 628", "NGC 1300", "M16", "M57", "Stephan's Quintet",
    "Jupiter", "Saturn", "Mars", "Europa", "Titan",
    "Sgr A*", "Crab Nebula", "Cartwheel Galaxy", "Pillars of Creation", "LMC",
]

FEATURED_TARGETS = [
    # Exoplanet hosts — good for timeline crossmatch
    ("WASP-76",      "Hot Jupiter, JWST atmosphere"),
    ("WASP-39",      "Hot Jupiter, JWST benchmark"),
    ("TRAPPIST-1",   "7 rocky planets"),
    ("HD 209458",    "First transiting exoplanet"),
    ("GJ 1214",      "Super-Earth / water world"),
    ("TOI-700",      "Habitable zone Earth-size"),
    ("LHS 1140",     "Rocky super-Earth"),
    ("K2-18",        "Sub-Neptune, possible ocean"),
    # Solar system
    ("Jupiter",      "Gas giant"),
    ("Saturn",       "Ringed planet"),
    ("Mars",         "Red planet"),
    ("Europa",       "Icy moon"),
    # Deep sky — good for JWST imaging
    ("NGC 628",      "Spiral galaxy, JWST flagship"),
    ("Crab Nebula",  "Supernova remnant"),
    ("M16",          "Pillars of Creation"),
    ("NGC 3132",     "Southern Ring Nebula"),
    ("Cartwheel Galaxy", "Ring galaxy, JWST"),
    ("NGC 1300",     "Barred spiral galaxy"),
    ("Stephan's Quintet", "Galaxy group, JWST"),
    ("Sgr A*",       "Milky Way black hole"),
]

@st.cache_data(show_spinner=False)
def resolve_target(target_str: str, mode: str):
    """Return a SkyCoord from a name or 'ra dec' string."""
    if mode == "Object name":
        # Check for solar system bodies first — SIMBAD won't have these
        if target_str.strip().lower() in SOLAR_SYSTEM_BODIES:
            from astropy.coordinates import get_body_barycentric
            from astropy.time import Time
            from astropy.coordinates import solar_system_ephemeris, ICRS
            import astropy.coordinates as coord_module
            with solar_system_ephemeris.set("builtin"):
                body = coord_module.get_body(target_str.strip().lower(), Time.now())
            coord = SkyCoord(ra=body.ra, dec=body.dec, frame="icrs")
            return coord

        # Try CDS Sesame (handles stars, galaxies, nebulae, exoplanet hosts)
        try:
            coord = _with_retry(lambda: SkyCoord.from_name(target_str))
            return coord
        except Exception:
            pass

        # Fallback: SIMBAD directly (column names vary by astroquery version)
        from astroquery.simbad import Simbad
        result = _with_retry(lambda: Simbad.query_object(target_str))
        if result is None or len(result) == 0:
            raise ValueError(
                f"Could not resolve '{target_str}'. "
                "Try an exact catalogue name (e.g. 'NGC 628', 'TRAPPIST-1') "
                "or switch to RA/Dec mode."
            )
        # Column names differ between astroquery versions: try both cases
        cols = {c.lower(): c for c in result.colnames}
        ra_col  = cols.get("ra",  cols.get("ra_d",  None))
        dec_col = cols.get("dec", cols.get("dec_d", None))
        if ra_col is None or dec_col is None:
            raise ValueError(f"Unexpected SIMBAD columns: {result.colnames}")
        ra_val  = result[ra_col][0]
        dec_val = result[dec_col][0]
        # SIMBAD returns sexagesimal strings for RA/Dec; degree columns end in _d
        if "d" in ra_col.lower():
            coord = SkyCoord(ra=float(ra_val), dec=float(dec_val), unit=u.deg, frame="icrs")
        else:
            coord = SkyCoord(ra=ra_val, dec=dec_val, unit=(u.hourangle, u.deg), frame="icrs")
    else:
        parts = target_str.split()
        if len(parts) != 2:
            raise ValueError("RA/Dec mode expects exactly two numbers, e.g. '261.7 -73.5'")
        coord = SkyCoord(ra=float(parts[0]), dec=float(parts[1]), unit=u.deg, frame="icrs")
    return coord


@st.cache_data(show_spinner=False)
def query_mast(ra_deg: float, dec_deg: float, radius_arcmin: float,
               mission: str, max_rows: int) -> pd.DataFrame:
    """Query MAST for observations near a coordinate."""
    coord = SkyCoord(ra=ra_deg, dec=dec_deg, unit=u.deg, frame="icrs")
    radius = u.Quantity(radius_arcmin, u.arcmin)
    obs = _with_retry(lambda: Observations.query_region(coord, radius=radius))
    if obs is None or len(obs) == 0:
        return pd.DataFrame()
    # Filter to the requested mission
    mask = [str(c).upper() == mission.upper() for c in obs["obs_collection"]]
    obs = obs[mask]
    if len(obs) == 0:
        return pd.DataFrame()

    df = obs.to_pandas()

    # Useful columns (keep only what's available)
    want = [
        "target_name", "obs_collection", "instrument_name", "filters",
        "t_exptime", "t_min", "t_max", "em_min", "em_max",
        "obs_id", "proposal_id", "dataURL", "jpegURL", "s_ra", "s_dec",
        "calib_level", "dataproduct_type",
    ]
    keep = [c for c in want if c in df.columns]
    df = df[keep].head(max_rows)

    # Human-readable date (t_min is MJD — use astropy Time to avoid overflow)
    if "t_min" in df.columns:
        from astropy.time import Time
        def mjd_to_date(val):
            try:
                return Time(float(val), format="mjd").to_datetime().date()
            except Exception:
                return None
        df["obs_date"] = df["t_min"].apply(mjd_to_date)

    return df


@st.cache_data(show_spinner=False)
def fetch_preview(url: str):
    """Download a JPEG preview image from MAST."""
    try:
        r = _with_retry(lambda: requests.get(url, timeout=10))
        r.raise_for_status()
        return Image.open(BytesIO(r.content))
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def search_lightcurve_meta(target_name: str) -> pd.DataFrame:
    """Return TESS light curve search results as a plain dataframe (cacheable)."""
    search = lk.search_lightcurve(target_name, mission="TESS", author="SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(target_name, mission="TESS")
    if len(search) == 0:
        return pd.DataFrame()
    return search.table.to_pandas()

def fetch_lightcurve(target_name: str):
    """Download the first available TESS light curve (not cached — lightkurve objects are not picklable)."""
    search = lk.search_lightcurve(target_name, mission="TESS", author="SPOC")
    if len(search) == 0:
        search = lk.search_lightcurve(target_name, mission="TESS")
    if len(search) == 0:
        return None
    return search[0].download()


@st.cache_data(show_spinner=False)
def get_ephemeris(target_name: str) -> dict | None:
    """Query NASA Exoplanet Archive for transit ephemeris.
    Returns dict with period, t0_mjd, duration_days, planet_name — or None."""
    try:
        from astroquery.ipac.nexsci.nasa_exoplanet_archive import NasaExoplanetArchive
        result = _with_retry(lambda: NasaExoplanetArchive.query_criteria(
            table="pscomppars",
            select="pl_name,hostname,pl_orbper,pl_tranmid,pl_trandur",
            where=f"hostname like '%{target_name}%'",
        ))
        if result is None or len(result) == 0:
            return None
        df = result.to_pandas().dropna(subset=["pl_orbper", "pl_tranmid", "pl_trandur"])
        if df.empty:
            return None
        row = df.iloc[0]
        return {
            "planet_name": row["pl_name"],
            "period":      float(row["pl_orbper"]),
            "t0_mjd":      float(row["pl_tranmid"]) - 2400000.5,  # BJD → MJD
            "duration":    float(row["pl_trandur"]) / 24.0,        # hours → days
        }
    except Exception:
        return None


def build_timeline(target: str, df_jwst: pd.DataFrame, df_tess: pd.DataFrame,
                   ephem: dict | None):
    """Render a JWST + TESS observation timeline with optional transit overlays."""
    import numpy as np
    import matplotlib.patches as mpatches
    from astropy.time import Time

    fig, ax = plt.subplots(figsize=(14, 3))

    # Collect all times to set axis range
    all_times = []

    # ── TESS bands ────────────────────────────────────────────────────────────
    if not df_tess.empty:
        tess_valid = df_tess.dropna(subset=["t_min", "t_max"])
        for _, row in tess_valid.iterrows():
            ax.axvspan(row["t_min"], row["t_max"],
                       ymin=0.55, ymax=0.95, color="steelblue", alpha=0.45)
            all_times += [row["t_min"], row["t_max"]]

    # ── JWST markers ─────────────────────────────────────────────────────────
    if not df_jwst.empty:
        jwst_valid = df_jwst.dropna(subset=["t_min"])
        for _, row in jwst_valid.iterrows():
            mid = (row["t_min"] + row.get("t_max", row["t_min"])) / 2
            ax.axvline(mid, ymin=0.05, ymax=0.50,
                       color="darkorange", linewidth=1.5, alpha=0.85)
            all_times.append(mid)

    # ── Transit windows ───────────────────────────────────────────────────────
    if ephem and all_times:
        window_start = min(all_times) - 30
        window_end   = max(all_times) + 30
        T0     = ephem["t0_mjd"]
        period = ephem["period"]
        half   = ephem["duration"] / 2

        n_start = int((window_start - T0) / period)
        n_end   = int((window_end   - T0) / period) + 1
        centers = T0 + np.arange(n_start, n_end) * period
        centers = centers[(centers >= window_start) & (centers <= window_end)]

        for tc in centers:
            ax.axvspan(tc - half, tc + half,
                       ymin=0.0, ymax=1.0, color="crimson", alpha=0.10, linewidth=0)
            ax.axvline(tc, color="crimson", linewidth=0.4, alpha=0.35)

    # ── Axes ──────────────────────────────────────────────────────────────────
    if all_times:
        margin = 30
        ax.set_xlim(min(all_times) - margin, max(all_times) + margin)

    # Year tick labels
    x_min, x_max = ax.get_xlim()
    year_start = int(Time(x_min, format="mjd").datetime.year)
    year_end   = int(Time(x_max, format="mjd").datetime.year) + 1
    year_ticks = {y: Time(f"{y}-01-01").mjd for y in range(year_start, year_end + 1)}
    ax.set_xticks(list(year_ticks.values()))
    ax.set_xticklabels(list(year_ticks.keys()))
    ax.set_yticks([])
    ax.set_xlabel("Year")

    title = f"{target} — Observation Timeline"
    if ephem:
        title += f"  ·  {ephem['planet_name']} transits overlaid"
    ax.set_title(title)

    # Row labels
    ax.text(0.005, 0.75, "TESS",  transform=ax.transAxes, fontsize=8, color="steelblue",  va="center")
    ax.text(0.005, 0.28, "JWST",  transform=ax.transAxes, fontsize=8, color="darkorange", va="center")

    legend = [
        mpatches.Patch(color="steelblue",  alpha=0.5, label="TESS sector"),
        mpatches.Patch(color="darkorange", alpha=0.8, label="JWST observation"),
    ]
    if ephem:
        legend.append(mpatches.Patch(color="crimson", alpha=0.25, label=f"{ephem['planet_name']} transit"))
    ax.legend(handles=legend, loc="upper right", fontsize=8)

    fig.tight_layout()
    return fig


# ── Featured targets (sidebar) ───────────────────────────────────────────────
with st.sidebar:
    st.divider()
    st.markdown("**Featured targets**")
    cols = st.columns(2)
    for i, (name, desc) in enumerate(FEATURED_TARGETS):
        if cols[i % 2].button(name, key=f"ft_{i}", help=desc, use_container_width=True):
            st.session_state.prefill_target = name
            st.rerun()

# ── Main app ──────────────────────────────────────────────────────────────────

if run_search:
    # Resolve coordinates
    with st.spinner("Resolving target…"):
        try:
            coord = resolve_target(target_input.strip(), search_mode)
        except Exception as e:
            st.error(f"Could not resolve target: {e}")
            st.stop()

    st.success(
        f"**{target_input}** → RA {coord.ra.deg:.4f}°, Dec {coord.dec.deg:.4f}°"
    )

    # Initialise result dataframes so timeline section can always reference them
    df_jwst = pd.DataFrame()
    df_tess = pd.DataFrame()

    # ── JWST tab ──────────────────────────────────────────────────────────────
    if "JWST" in missions:
        st.subheader("🪐 JWST Observations")
        with st.spinner("Querying MAST for JWST…"):
            df_jwst = query_mast(
                coord.ra.deg, coord.dec.deg,
                radius_arcmin, "JWST", max_results,
            )

        if df_jwst.empty:
            st.info("No JWST observations found for this target / radius.")
        else:
            st.caption(f"{len(df_jwst)} observations returned")

            # Display columns
            display_cols = [c for c in [
                "target_name", "instrument_name", "filters",
                "t_exptime", "obs_date", "proposal_id", "dataproduct_type", "calib_level",
            ] if c in df_jwst.columns]

            st.dataframe(
                df_jwst[display_cols],
                use_container_width=True,
                hide_index=True,
            )

            # Image previews
            preview_rows = df_jwst[df_jwst["jpegURL"].notna()] if "jpegURL" in df_jwst.columns else pd.DataFrame()
            if not preview_rows.empty:
                st.markdown("#### Image Previews")
                cols = st.columns(4)
                shown = 0
                for _, row in preview_rows.iterrows():
                    if shown >= 8:
                        break
                    img = fetch_preview(row["jpegURL"])
                    if img:
                        label = f"{row.get('target_name','?')} · {row.get('filters','')}"
                        cols[shown % 4].image(img, caption=label, use_container_width=True)
                        shown += 1
                if shown == 0:
                    st.info("Preview images not available for these observations.")

            # Build MAST deep links using the confirmed URL structure
            # e.g. jw05924-o015_... → program_id=5924&obs_id=15
            if "obs_id" in df_jwst.columns:
                import re
                st.markdown("#### View on MAST")
                for _, row in df_jwst.head(10).iterrows():
                    oid = row["obs_id"]
                    target = row.get("target_name", "").strip()
                    filters = row.get("filters", "")
                    prog_match = re.match(r"jw(\d+)", oid)
                    obs_match  = re.search(r"-o(\d+)", oid)
                    if prog_match and obs_match:
                        prog_id = str(int(prog_match.group(1)))
                        obs_num = str(int(obs_match.group(1)))
                        import urllib.parse
                        target_enc = urllib.parse.quote(target)
                        results_url = (
                            "https://mast.stsci.edu/search/ui/#/jwst/results"
                            "?resolve=true"
                            "&data_types=spectrum,timeseries,image,other"
                            "&instruments=MIRI,NIRCAM,NIRSPEC,NIRISS,FGS"
                            f"&target_name={target_enc}"
                            f"&program_id={prog_id}"
                            f"&obs_id={obs_num}"
                            "&useStore=false"
                        )
                        st.markdown(
                            f"- **{target}** `{filters}` "
                            f"Program **{prog_id}** Obs **{obs_num}** "
                            f"→ [Results]({results_url})"
                        )
                    else:
                        # Fallback for non-standard obs_id formats
                        st.markdown(f"- **{target}** &nbsp;`{filters}` &nbsp;`{oid}`")

    # ── TESS tab ──────────────────────────────────────────────────────────────
    if "TESS" in missions:
        st.subheader("⭐ TESS Observations")

        with st.spinner("Querying MAST for TESS…"):
            df_tess = query_mast(
                coord.ra.deg, coord.dec.deg,
                radius_arcmin, "TESS", max_results,
            )

        if df_tess.empty:
            st.info("No TESS observations found for this target / radius.")
        else:
            display_cols = [c for c in [
                "target_name", "instrument_name", "filters",
                "t_exptime", "obs_date", "proposal_id", "dataproduct_type", "calib_level",
            ] if c in df_tess.columns]
            st.caption(f"{len(df_tess)} observations returned")
            st.dataframe(df_tess[display_cols], use_container_width=True, hide_index=True)

        # Light curve
        st.markdown("#### Light Curve (TESS via lightkurve)")
        with st.spinner("Searching for light curves…"):
            try:
                lc_meta = search_lightcurve_meta(target_input.strip())
            except Exception as e:
                lc_meta = pd.DataFrame()
                st.warning(f"lightkurve search error: {e}")

        if not lc_meta.empty:
            with st.expander("Light curve search results"):
                st.dataframe(lc_meta, use_container_width=True, hide_index=True)

        lc = None
        with st.spinner("Downloading light curve…"):
            try:
                lc = fetch_lightcurve(target_input.strip())
            except Exception as e:
                st.warning(f"lightkurve download error: {e}")

        if lc is not None:

            fig, ax = plt.subplots(figsize=(10, 3))
            lc.scatter(ax=ax, s=1, c="steelblue")
            ax.set_title(f"TESS light curve — {target_input}")
            ax.set_xlabel("Time (BTJD)")
            ax.set_ylabel("Flux")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        else:
            st.info(
                "No TESS light curve found via lightkurve for this target name. "
                "Try an exact catalogue name (e.g. 'TRAPPIST-1' or 'TIC 278683025')."
            )

    # ── Timeline crossmatch ───────────────────────────────────────────────────
    # Show when both missions are selected and at least one has results
    df_jwst_tl = df_jwst if "JWST" in missions and not df_jwst.empty else pd.DataFrame()
    df_tess_tl = df_tess if "TESS" in missions and not df_tess.empty else pd.DataFrame()

    if not df_jwst_tl.empty or not df_tess_tl.empty:
        st.subheader("📅 Observation Timeline")

        with st.spinner("Checking NASA Exoplanet Archive…"):
            # Strip trailing planet designations (e.g. "WASP-76 b" → "WASP-76")
            # and use the MAST target_name as a fallback if it differs from user input
            host_candidates = set()
            host_candidates.add(target_input.strip())
            # Add version with planet letter stripped
            import re as _re
            stripped = _re.sub(r"\s+[a-zA-Z]$", "", target_input.strip()).strip()
            host_candidates.add(stripped)
            # Also try the most common target_name from MAST results
            for _df in [df_jwst_tl, df_tess_tl]:
                if not _df.empty and "target_name" in _df.columns:
                    top = _df["target_name"].dropna().mode()
                    if len(top):
                        host_candidates.add(top.iloc[0].strip())

            ephem = None
            ephem_query_used = None
            for candidate in host_candidates:
                ephem = get_ephemeris(candidate)
                if ephem:
                    ephem_query_used = candidate
                    break

        if ephem:
            st.caption(
                f"Found ephemeris for **{ephem['planet_name']}** "
                f"(queried as `{ephem_query_used}`) — "
                f"period {ephem['period']:.4f} d, "
                f"transit duration {ephem['duration']*24:.2f} h"
            )
        else:
            tried = ", ".join(f"`{c}`" for c in host_candidates)
            st.caption(f"No exoplanet ephemeris found (tried: {tried}) — showing observation timeline only.")

        try:
            fig = build_timeline(target_input.strip(), df_jwst_tl, df_tess_tl, ephem)
            st.pyplot(fig)
            plt.close(fig)
        except Exception as e:
            st.warning(f"Could not render timeline: {e}")

else:
    st.info("Enter a target in the sidebar and press **Search** to begin.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Data from [MAST](https://mast.stsci.edu) · "
    "Powered by [astroquery](https://astroquery.readthedocs.io) & "
    "[lightkurve](https://lightkurve.github.io/lightkurve/)"
)
