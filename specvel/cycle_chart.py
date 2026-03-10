"""
specvel/cycle_chart.py

Visual output for the spectral velocity cycle analysis.

Charts show:
  1. Normalized price series colored by cycle phase
  2. Velocity series with expected-velocity band shaded
  3. Velocity surprise (current - expected) as bar chart
  4. Phase timeline at bottom

Usage:
    from cycle import compute_velocity_surprise
    from cycle_chart import plot_cycle_analysis, plot_cycle_dashboard

    surp = compute_velocity_surprise(normed_series, cycle_method='auto')
    plot_cycle_analysis(normed_series, surp, title='S&P 500')
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap

from core import compute_velocity
from cycle import (
    PHASE_GREEN_UP, PHASE_PEAK, PHASE_SENESCENCE, PHASE_DORMANT,
    compute_velocity_surprise,
)


# ── Phase color palette ───────────────────────────────────────────────────────
PHASE_COLORS = {
    PHASE_GREEN_UP:   "#2ecc71",   # green
    PHASE_PEAK:       "#f39c12",   # amber
    PHASE_SENESCENCE: "#e74c3c",   # red
    PHASE_DORMANT:    "#95a5a6",   # grey
}

PHASE_LABELS = {
    PHASE_GREEN_UP:   "Green-up (Early Expansion)",
    PHASE_PEAK:       "Peak (Late Expansion)",
    PHASE_SENESCENCE: "Senescence (Contraction)",
    PHASE_DORMANT:    "Dormant (Trough)",
}


def plot_cycle_analysis(
    series:      pd.Series,
    surp:        dict,
    title:       str  = "",
    figsize:     tuple = (14, 10),
    save_path:   str  = None,
    show:        bool = True,
) -> plt.Figure:
    """
    Four-panel cycle analysis chart for a single series.

    Panel 1: Price colored by cycle phase
    Panel 2: Velocity + expected velocity band
    Panel 3: Velocity surprise bars
    Panel 4: Phase timeline
    """
    vel     = surp.get("_vel", pd.Series(dtype=float))
    phases  = surp.get("_phases", pd.Series(dtype=str))
    base    = surp.get("_baseline", {})

    s       = series.dropna()
    v       = vel.dropna()
    ph      = phases.reindex(v.index).fillna(PHASE_DORMANT)

    if s.empty or v.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Insufficient data", ha='center', va='center')
        return fig

    fig = plt.figure(figsize=figsize)
    gs  = fig.add_gridspec(4, 1, height_ratios=[3, 2.5, 2, 0.6], hspace=0.08)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    ax3 = fig.add_subplot(gs[2], sharex=ax1)
    ax4 = fig.add_subplot(gs[3], sharex=ax1)

    # ── Panel 1: Price colored by phase ──────────────────────────────────────
    s_aligned = s.reindex(v.index).dropna()
    prev_phase = None
    seg_start  = None
    seg_x, seg_y, seg_c = [], [], []

    idx_list = list(s_aligned.index)
    for i, idx in enumerate(idx_list):
        p = ph.get(idx, PHASE_DORMANT)
        if p != prev_phase:
            if prev_phase is not None and seg_x:
                ax1.plot(seg_x, seg_y, color=PHASE_COLORS[prev_phase],
                         linewidth=2.0, solid_capstyle='round')
            seg_x, seg_y = [], []
            prev_phase = p
        seg_x.append(idx)
        seg_y.append(s_aligned[idx])
    if seg_x:
        ax1.plot(seg_x, seg_y, color=PHASE_COLORS.get(prev_phase, 'gray'),
                 linewidth=2.0, solid_capstyle='round')

    ax1.set_ylabel("Price (normalized)", fontsize=9)
    ax1.set_title(
        f"{title}\nSpectral Velocity Cycle Analysis  "
        f"│  Current phase: {surp.get('current_phase','').upper()}  "
        f"│  Age: {surp.get('phase_age',0)} periods  "
        f"│  {surp.get('surprise_signal','')}",
        fontsize=11, fontweight='bold', loc='left'
    )

    # Phase legend
    handles = [mpatches.Patch(color=PHASE_COLORS[p], label=PHASE_LABELS[p])
               for p in [PHASE_GREEN_UP, PHASE_PEAK, PHASE_SENESCENCE, PHASE_DORMANT]]
    ax1.legend(handles=handles, loc='upper left', fontsize=7.5,
               ncol=2, framealpha=0.85)

    # ── Panel 2: Velocity + expected band ────────────────────────────────────
    ax2.plot(v.index, v.values, color='#2c3e50', linewidth=1.3,
             label='Velocity', zorder=3)
    ax2.axhline(0, color='black', linewidth=0.5, linestyle='--', alpha=0.4)

    # Shade expected velocity range per phase
    for phase, color in PHASE_COLORS.items():
        mask = ph == phase
        if mask.any():
            p_base = base.get(phase, {})
            mean   = p_base.get("mean", 0)
            p25    = p_base.get("p25", mean - 0.05)
            p75    = p_base.get("p75", mean + 0.05)
            phase_idx = v.index[mask.reindex(v.index, fill_value=False)]
            ax2.fill_between(phase_idx,
                             [p25] * len(phase_idx),
                             [p75] * len(phase_idx),
                             alpha=0.18, color=color,
                             label=f"Expected ({phase[:4]})")
            ax2.plot(phase_idx, [mean] * len(phase_idx),
                     color=color, linewidth=0.8, linestyle=':', alpha=0.7)

    # Mark current expected vs actual
    exp_vel = surp.get("expected_velocity", 0)
    cur_vel = surp.get("current_velocity", 0)
    last_date = v.index[-1]
    ax2.annotate(f"Expected: {exp_vel:+.4f}",
                 xy=(last_date, exp_vel),
                 xytext=(-80, 10), textcoords='offset points',
                 fontsize=8, color='gray',
                 arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))
    ax2.annotate(f"Actual: {cur_vel:+.4f}",
                 xy=(last_date, cur_vel),
                 xytext=(-80, -15), textcoords='offset points',
                 fontsize=8, color='#2c3e50',
                 arrowprops=dict(arrowstyle='->', color='#2c3e50', lw=0.8))

    ax2.set_ylabel("Velocity", fontsize=9)
    ax2.legend(loc='upper left', fontsize=7, ncol=3, framealpha=0.8)

    # ── Panel 3: Velocity surprise bars ──────────────────────────────────────
    # Surprise = actual vel - expected vel for current phase at each point
    surprise_series = pd.Series(dtype=float)
    for phase in [PHASE_GREEN_UP, PHASE_PEAK, PHASE_SENESCENCE, PHASE_DORMANT]:
        mask    = ph == phase
        p_mean  = base.get(phase, {}).get("mean", 0)
        phase_v = v[mask.reindex(v.index, fill_value=False)]
        surp_ph = phase_v - p_mean
        surprise_series = pd.concat([surprise_series, surp_ph])
    surprise_series = surprise_series.sort_index()

    colors = ['#2ecc71' if x >= 0 else '#e74c3c'
              for x in surprise_series.values]
    ax3.bar(surprise_series.index, surprise_series.values,
            color=colors, alpha=0.75, width=pd.Timedelta(days=2))
    ax3.axhline(0, color='black', linewidth=0.6)

    # Threshold lines
    sz = surp.get("surprise_zscore", 0)
    ax3.set_ylabel("Velocity Surprise\n(actual − expected)", fontsize=8)

    # Annotate current surprise
    surp_val = surp.get("velocity_surprise", 0)
    ax3.text(0.99, 0.92,
             f"Current surprise: {surp_val:+.4f}  (z={sz:+.2f})",
             transform=ax3.transAxes, fontsize=8.5, ha='right',
             color='#2ecc71' if surp_val >= 0 else '#e74c3c',
             fontweight='bold')

    # Transition warning
    warn = surp.get("transition_warning", "")
    if warn:
        ax3.text(0.01, 0.08, warn, transform=ax3.transAxes,
                 fontsize=8, color='#e67e22', style='italic')

    # ── Panel 4: Phase timeline ───────────────────────────────────────────────
    prev_p    = None
    seg_start = None
    for i, idx in enumerate(ph.index):
        p = ph.iloc[i]
        if p != prev_p:
            if prev_p is not None and seg_start is not None:
                ax4.axvspan(seg_start, idx, alpha=0.5,
                            color=PHASE_COLORS.get(prev_p, 'gray'))
            seg_start = idx
            prev_p = p
    if prev_p and seg_start:
        ax4.axvspan(seg_start, ph.index[-1], alpha=0.5,
                    color=PHASE_COLORS.get(prev_p, 'gray'))

    ax4.set_yticks([])
    ax4.set_ylabel("Phase", fontsize=8)
    ax4.xaxis.set_major_formatter(mdates.DateFormatter('%b %y'))
    plt.setp(ax4.xaxis.get_majorticklabels(), rotation=30, fontsize=8)

    for ax in [ax1, ax2, ax3]:
        plt.setp(ax.xaxis.get_majorticklabels(), visible=False)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()

    return fig


def plot_cycle_dashboard(
    df:       pd.DataFrame,
    adapter,
    start:    str,
    end:      str,
    top_n:    int   = 6,
    figsize:  tuple = (16, 14),
    save_path: str  = None,
    show:     bool  = True,
) -> plt.Figure:
    """
    Multi-panel dashboard showing top N series by surprise z-score.
    Each mini-panel shows: phase-colored price + velocity surprise.
    """
    top = df.head(top_n)
    n   = len(top)
    cols = 2
    rows = (n + 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.array(axes).flatten()

    cycle_method = getattr(adapter, 'CYCLE_METHOD', 'auto')

    for i, (_, row) in enumerate(top.iterrows()):
        ax = axes[i]
        sid = row['series_id']

        try:
            raw    = adapter.fetch(sid, start, end)
            normed = adapter.normalize(raw)
            surp   = compute_velocity_surprise(normed, cycle_method=cycle_method)

            vel    = surp["_vel"].dropna()
            phases = surp["_phases"].reindex(vel.index).fillna(PHASE_DORMANT)
            s      = normed.reindex(vel.index).dropna()

            # Price line colored by phase
            prev_p = None
            seg_x, seg_y = [], []
            for idx in s.index:
                p = phases.get(idx, PHASE_DORMANT)
                if p != prev_p:
                    if prev_p and seg_x:
                        ax.plot(seg_x, seg_y, color=PHASE_COLORS[prev_p],
                                lw=1.5)
                    seg_x, seg_y, prev_p = [], [], p
                seg_x.append(idx)
                seg_y.append(s[idx])
            if seg_x:
                ax.plot(seg_x, seg_y,
                        color=PHASE_COLORS.get(prev_p, 'gray'), lw=1.5)

            # Velocity on twin axis
            ax2 = ax.twinx()
            ax2.plot(vel.index, vel.values, color='#2c3e50',
                     lw=0.9, alpha=0.6)
            ax2.axhline(0, color='black', lw=0.4, ls='--')

            # Shade surprise direction
            exp   = surp["expected_velocity"]
            ax2.fill_between(vel.index, vel.values, exp,
                             where=(vel.values >= exp),
                             alpha=0.2, color='green', label='Above expected')
            ax2.fill_between(vel.index, vel.values, exp,
                             where=(vel.values < exp),
                             alpha=0.2, color='red', label='Below expected')

            sz      = surp["surprise_zscore"]
            phase   = surp["current_phase"]
            sig     = surp["surprise_signal"]
            conv    = row.get("conviction", 0)
            boost   = row.get("conviction_boost", 0)
            label   = row.get("label", sid)[:20]
            warn    = surp.get("transition_warning", "")

            color   = '#2ecc71' if conv > 0 else ('#e74c3c' if conv < 0 else '#95a5a6')
            ax.set_title(
                f"{label}\n"
                f"{phase.upper()}  │  z={sz:+.2f}  │  conv={conv:+d} "
                f"(base+{boost:+d})\n"
                f"{sig[:30]}" +
                (f"\n{warn[:35]}" if warn else ""),
                fontsize=8, color=color, fontweight='bold', loc='left'
            )
            ax.set_ylabel("Price", fontsize=7)
            ax2.set_ylabel("Vel", fontsize=7, color='#2c3e50')
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b%y'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=25, fontsize=7)

        except Exception as e:
            ax.set_title(f"{sid}\nError: {str(e)[:40]}", fontsize=8)

    # Hide unused panels
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    # Phase legend at bottom
    handles = [mpatches.Patch(color=PHASE_COLORS[p], label=PHASE_LABELS[p])
               for p in [PHASE_GREEN_UP, PHASE_PEAK, PHASE_SENESCENCE, PHASE_DORMANT]]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=8.5, framealpha=0.9,
               bbox_to_anchor=(0.5, 0.01))

    fig.suptitle(
        f"Spectral Velocity Cycle Dashboard — {adapter.source_name.upper()}\n"
        f"Top {n} by |Surprise Z-Score|  │  Method: {cycle_method}",
        fontsize=12, fontweight='bold', y=1.01
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    if show:
        plt.show()

    return fig
