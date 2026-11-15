import json, math, os, re, queue, threading, asyncio
from pathlib import Path
import numpy as np
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
from scipy.ndimage import distance_transform_edt

HERE = Path(__file__).resolve().parent

# ------------------------------------------------------------------ LLM settings (from SARA main.py)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
MODELS = ["openai/gpt-oss-20b", "openai/gpt-oss-20b"]   # second is a fallback
VOICE = "en-GB-AriaNeural"

SYSTEM_PROMPT = (
    "You are SARA, the Science and Robotics Assistant, acting as a spacecraft fire-safety risk analyst. "
    "You receive JSON output from reduced-order fire simulations (flame growth or droplet burning, smoke transport, "
    "detector alarm times) driven by a flame detection system and a smoke sensor. "
    "Write a risk assessment report in clean plain text: no markdown, no asterisks, no emoji, no LaTeX. "
    "Use these headings in capitals, each on its own line: "
    "OVERALL RISK (one of LOW, MODERATE, HIGH, CRITICAL, with a one sentence reason), SITUATION, FIRE GROWTH, "
    "SMOKE AND DETECTION, CABIN IMPACT, RECOMMENDED ACTIONS (numbered, most urgent first), CONFIDENCE AND LIMITS. "
    "Use only numbers that appear in the data. Quote low / mid / high ranges where given. Do not invent values. "
    "If a detector has no alarm time, say it did not alarm in the simulated window. "
    "Mention the smoke sensor flame type and how it shaped the result, and carry over the notes in the data. "
    "State clearly that this is a screening model, not a safety certification. Keep the report under 450 words."
)

# ------------------------------------------------------------------ constants
G0, T_INF = 9.80665, 298.15
K_HOT, CP_HOT = 0.07, 1300.0           # ASSUMPTION: hot gas film around a droplet (W/mK, J/kgK)
K_FILM = 0.05                          # ASSUMPTION: gas conductivity in the de Ris spread formula
RHO_AIR, CP_AIR, ALPHA_AIR = 1.18, 1005.0, 2.2e-5
ETA_FLAMING, ETA_SMOULDER = 0.9, 0.3   # combustion efficiency
CABIN = dict(lx=4.0, ly=2.0, depth=2.0)

FUELS = {
    # kind, LHV J/kg, s = kg O2 per kg fuel, soot = smoke yield kg/kg when flaming
    "PMMA acrylic sheet": dict(kind="solid", rho_s=1190, cp_s=1466, k_s=0.19, T_v=660, T_f21=1800, LHV=24.9e6,
                               s=1.918, soot=0.022, thick_mm=0.1, sides=2, regress=0.010, o2_limit=0.15),
    "Cotton fabric":      dict(kind="solid", rho_s=400, cp_s=1300, k_s=0.15, T_v=620, T_f21=1600, LHV=16.1e6,
                               s=1.184, soot=0.008, thick_mm=0.3, sides=2, regress=0.030, o2_limit=0.17),
    "n-Heptane (liquid)": dict(kind="liquid", LHV=44.6e6, s=3.513, h_fg=0.317e6, T_boil=371.6, rho_l=684.0,
                               soot=0.037, o2_limit=0.16),
    "Methanol (liquid)":  dict(kind="liquid", LHV=19.9e6, s=1.498, h_fg=1.100e6, T_boil=337.8, rho_l=792.0,
                               soot=0.001, o2_limit=0.16),
}
RUNS = ("lo", "mid", "hi")
SMOKE_TYPES = ["Flaming", "Smouldering", "Unknown"]


def y_o2(x):                            # O2 mass fraction from mole fraction
    return 32 * x / (32 * x + 28.01 * (1 - x))


def sig(x, n=3):
    return float(f"{x:.{n}g}")


# ------------------------------------------------------------------ liquids: d-squared law (sim1 A, sim5 A)
def liquid_fire(f, p, notes):
    x = p["o2"] / 100
    B = (y_o2(x) * f["LHV"] / f["s"] + CP_HOT * (T_INF - f["T_boil"])) / f["h_fg"]
    K0 = 8 * K_HOT / (f["rho_l"] * CP_HOT) * math.log(1 + B)               # m2/s (textbook)
    r_f = (3 * p["volume_cm3"] / (4 * math.pi)) ** (1 / 3)                  # cm, equivalent flame radius
    d0_mm = min(8.0, max(0.3, 2 * r_f / 6 * 10))                            # ASSUMPTION: flame diameter ~ 6 x droplet
    if abs(d0_mm - 2 * r_f / 6 * 10) > 1e-9:
        notes.append(f"Droplet size clipped to {d0_mm:.1f} mm; the flame is bigger than one droplet would explain.")
    notes.append("Droplet size inferred from flame volume (flame diameter about 6x droplet diameter). "
                 "Textbook burning rate usually over-predicts NASA FLEX data, so the low run uses 0.6x K.")
    d0 = d0_mm / 1000
    ks = dict(lo=0.6, mid=1.0, hi=1.3)
    burn = {r: d0 ** 2 / (K0 * ks[r]) for r in RUNS}
    t_end = max(burn.values())
    times = np.linspace(0, t_end, 400)
    m, dia = {}, {}
    for r in RUNS:
        K = K0 * ks[r]
        d = np.sqrt(np.clip(d0 ** 2 - K * times, 0, None))
        dia[r] = d * 1000
        m[r] = math.pi / 4 * f["rho_l"] * K * d                              # kg/s
    info = dict(model="droplet d-squared law", droplet_mm=sig(d0_mm), K_mm2_s=sig(K0 * 1e6),
                burnout_s={r: sig(burn[r]) for r in RUNS}, transfer_number_B=sig(B))
    panel = ("Droplet diameter", "mm", dia)
    return times, m, panel, info


# ------------------------------------------------------------------ solids: de Ris spread + Huygens growth (sim1 B, sim4)
def solid_fire(f, p, smoulder, notes):
    x, g = p["o2"] / 100, (G0 if p["gravity"].startswith("1") else 0.0)
    thick = p["thick_mm"] / 1000
    tau = thick / 2 if f["sides"] == 2 else thick
    Tf = T_INF + (f["T_f21"] - T_INF) * y_o2(x) / y_o2(0.21)
    F = (Tf - f["T_v"]) / (f["T_v"] - T_INF)
    Vg = p["flow"] / 100 + ((g * ALPHA_AIR * (Tf - T_INF) / T_INF) ** (1 / 3) if g > 0 else 0.0)   # opposed flow, m/s
    thin = math.pi / 4 * K_FILM / (f["rho_s"] * f["cp_s"] * tau) * F
    thick_v = Vg * (K_FILM * RHO_AIR * CP_AIR) / (f["k_s"] * f["rho_s"] * f["cp_s"]) * F ** 2
    v0 = max(thin, thick_v)                                                  # m/s
    regime = "thin" if thin >= thick_v else "thick"
    limit = None
    if x < f["o2_limit"]:
        v0, limit = 0.0, f"oxygen {p['o2']:.1f}% is below the screening limit of {f['o2_limit'] * 100:.0f}%"
    elif Vg < 0.01:
        v0, limit = 0.0, "airflow below about 1 cm/s: flame quenches (no oxygen reaches it)"
    elif Vg > 0.40:
        v0, limit = 0.0, "airflow above about 40 cm/s: flame blows off"
    if limit:
        notes.append("Spread stopped: " + limit + ". (Limits scaled from thin PMMA BASS-II points.)")
    if smoulder:
        v0 *= 0.25                                                            # ASSUMPTION
        notes.append("Smouldering: spread x0.25, burn time x3, combustion efficiency 0.3, smoke yield x5 (assumptions).")
    tb = (tau * 1000) / f["regress"] * (3 if smoulder else 1)                # s, burnout time of a cell
    # Huygens growth on a grid over the fuel patch
    L, W = p["patch_L"], p["patch_W"]
    dx0 = min(0.25, L / 200, W / 4)
    nx, ny = math.ceil(L / dx0), math.ceil(W / dx0); dx, dy = L / nx, W / ny
    X, Y = np.meshgrid((np.arange(nx) + .5) * dx, (np.arange(ny) + .5) * dy, indexing="ij")
    r_fl = (3 * p["volume_cm3"] / (4 * math.pi)) ** (1 / 3)                  # cm, flame footprint radius
    init = (X - L / 2) ** 2 + (Y - W / 2) ** 2 <= r_fl ** 2
    if not init.any():
        init[nx // 2, ny // 2] = True
    cell = dx * dy
    dist = distance_transform_edt(~init, sampling=(dx, dy))                   # cm to nearest lit cell
    arrivals = {}
    for r, fac in zip(RUNS, (0.5, 1.0, 2.0)):
        v = v0 * fac * 100                                                    # cm/s
        t = np.full(init.shape, np.inf); t[init] = 0.0
        if v > 0:
            t[~init] = dist[~init] / v
        arrivals[r] = np.sort(t[np.isfinite(t)])
    t_end = min(900.0, max(a[-1] + tb for a in arrivals.values()))
    times = np.linspace(0, t_end, 500)
    e_area = f["rho_s"] * thick                                               # kg per m2 of fuel
    m, area = {}, {}
    for r in RUNS:
        lit = arrivals[r]
        a = (np.searchsorted(lit, times, "right") - np.searchsorted(lit, times - tb, "right")) * cell
        area[r] = a
        m[r] = a / 1e4 * e_area / tb                                          # kg/s
    stop = "reached the fuel edge" if len(arrivals["mid"]) == nx * ny else ("spread stopped by limit" if limit else "fire stopped before the edge")
    info = dict(model="de Ris spread + Huygens growth", regime=regime, flame_temp_K=sig(Tf),
                spread_mm_s={r: sig(v0 * fac * 1000) for r, fac in zip(RUNS, (0.5, 1, 2))},
                cell_burnout_s=sig(tb), initial_burning_cm2=sig(float(init.sum() * cell)),
                fuel_patch_cm2=sig(L * W), max_area_cm2={r: sig(float(area[r].max())) for r in RUNS},
                stop_reason=stop, effective_opposed_flow_cm_s=sig(Vg * 100))
    panel = ("Burning area", "cm$^2$", area)
    return times, m, panel, info


# ------------------------------------------------------------------ smoke drift grid (sim2)
NX, NY = 60, 30
DX = CABIN["lx"] / NX
SUPPLY = (int(round(0.4 / 3 / DX)), int(round(2.0 / 3 / DX)))
RETURN = (int(round(0.8 / 3 / DX)), int(round(0.8 / DX)))
DETECTORS = {"Ceiling, mid-cabin (Earth habit)": (NX // 2, NY - 1),
             "At the air return": (NX - 1, (RETURN[0] + RETURN[1]) // 2)}
ALARM_MG_M3 = 1.0


def load_cal():
    """Use the repo's tuned Saffire-III values if the folder is nearby, else documented defaults."""
    for base in (HERE, HERE.parent, HERE / "Vulcan_Orbit_v1", HERE.parent / "Vulcan_Orbit_v1"):
        fp = base / "data" / "drift_calibration.json"
        if fp.exists():
            try:
                c = json.loads(fp.read_text())
                return (max(c["D_values_kept"]), math.sqrt(c["clearance_s_range"][0] * c["clearance_s_range"][1]),
                        float(np.mean(c["delay_s_range"])), "Saffire-III calibration from repo")
            except Exception:
                pass
    return 0.01, 300.0, 20.0, "default values (ASSUMPTION: D 0.01 m2/s, clearance 300 s, delay 20 s)"


def air_field(flow):
    u = np.zeros((NY, NX + 1)); v = np.zeros((NY + 1, NX))
    n_in, n_out = SUPPLY[1] - SUPPLY[0], RETURN[1] - RETURN[0]
    v[NY, SUPPLY[0]:SUPPLY[1]] = -1.0
    u[RETURN[0]:RETURN[1], NX] = n_in / n_out
    b = ((u[:, 1:] - u[:, :-1] + v[1:, :] - v[:-1, :]) / DX).ravel()
    n = NX * NY; idx = np.arange(n); i, j = idx % NX, idx // NX
    nb = (i > 0).astype(float) + (i < NX - 1) + (j > 0) + (j < NY - 1)
    A = diags([-nb, (i < NX - 1)[:-1].astype(float), (i > 0)[1:].astype(float), np.ones(n - NX), np.ones(n - NX)],
              [0, 1, -1, NX, -NX], format="lil")
    rhs = -b * DX * DX
    A[0, :] = 0; A[0, 0] = 1; rhs[0] = 0
    phi = spsolve(csr_matrix(A), rhs).reshape(NY, NX)
    u[:, 1:-1] = (phi[:, 1:] - phi[:, :-1]) / DX
    v[1:-1, :] = (phi[1:, :] - phi[:-1, :]) / DX
    mean = np.hypot((u[:, 1:] + u[:, :-1]) / 2, (v[1:, :] + v[:-1, :]) / 2).mean()
    return u / mean * flow, v / mean * flow, n_in * DX * CABIN["depth"] / mean * flow


def vanleer(c, axis):
    d = np.diff(c, axis=axis)
    z = np.zeros_like(c.take([0], axis=axis))
    a, b = np.concatenate([z, d], axis), np.concatenate([d, z], axis)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(a * b > 0, 2 * a * b / (a + b), 0.0)


def smoke_run(rate_fn, fire_xy, flow, t_end=300.0):
    D, clear_s, delay, cal_note = load_cal()
    FIRE = (min(NX - 1, max(0, int(fire_xy[0] / DX))), min(NY - 1, max(0, int(fire_xy[1] / DX))))
    u, v, q = air_field(flow)
    vol = CABIN["lx"] * CABIN["ly"] * CABIN["depth"]
    removal = min(1.0, vol / q / clear_s)
    umax = (np.abs(u[:, 1:]) + np.abs(v[1:, :])).max()
    dt = 1.0 / math.ceil(1.0 / min(0.25 * DX / umax, 0.8 * 0.25 * DX * DX / D, 1.0))
    sps = round(1 / dt)
    c = np.zeros((NY, NX)); csup = 0.0
    probes = {k: [] for k in DETECTORS}; frames = {}
    for k in range(int(round(t_end / dt)) + 1):
        if k % sps == 0:
            for name, (pi, pj) in DETECTORS.items():
                probes[name].append(c[max(0, pj - 1):pj + 2, max(0, pi - 1):pi + 2].mean())
            t = k // sps
            if t in (60, 120, 300):
                frames[t] = c.copy()
        sx, sy = vanleer(c, 1), vanleer(c, 0)
        w = u[:, 1:-1]; h = 0.5 * (1 - np.abs(w) * dt / DX)
        cf = np.where(w > 0, c[:, :-1] + h * sx[:, :-1], c[:, 1:] - h * sx[:, 1:])
        fx = np.zeros((NY, NX + 1)); fx[:, 1:-1] = w * cf - D * np.diff(c, axis=1) / DX
        fx[:, -1] = u[:, -1] * np.where(u[:, -1] > 0, c[:, -1], csup)
        w = v[1:-1, :]; h = 0.5 * (1 - np.abs(w) * dt / DX)
        cf = np.where(w > 0, c[:-1, :] + h * sy[:-1, :], c[1:, :] - h * sy[1:, :])
        fy = np.zeros((NY + 1, NX)); fy[1:-1, :] = w * cf - D * np.diff(c, axis=0) / DX
        fy[-1, :] = v[-1, :] * np.where(v[-1, :] > 0, c[-1, :], csup)
        out = (fx[:, -1] * (u[:, -1] > 0)).sum(); qout = u[:, -1][u[:, -1] > 0].sum()
        c -= dt * (np.diff(fx, axis=1) + np.diff(fy, axis=0)) / DX
        c[FIRE[1], FIRE[0]] += rate_fn(k * dt) * dt / (DX * DX * CABIN["depth"])
        csup = (1 - removal) * out / qout
    times = np.arange(len(probes[next(iter(probes))]))
    alarms, peaks = {}, {}
    for name, s in probes.items():
        s = np.array(s); probes[name] = s; peaks[name] = float(s.max())
        hit = np.nonzero(s >= ALARM_MG_M3)[0]
        if len(hit) == 0:
            alarms[name] = None
        else:
            k = hit[0]
            t0 = times[k] if k == 0 else times[k - 1] + (ALARM_MG_M3 - s[k - 1]) / (s[k] - s[k - 1])
            alarms[name] = float(t0 + delay)
    return dict(alarms=alarms, peaks=peaks, times=times, probes=probes, frames=frames, fire_cell=FIRE,
                cal=cal_note, D=D, delay=delay)


# ------------------------------------------------------------------ run everything
def run_all(p):
    f = FUELS[p["fuel"]]; notes = []
    smoulder = p["smoke_type"] == "Smouldering"
    if smoulder and f["kind"] == "liquid":
        notes.append("Smoke sensor says smouldering but liquid fuels do not smoulder; treated as flaming.")
        smoulder = False
    if p["smoke_type"] == "Unknown":
        notes.append("Smoke sensor could not classify the flame; flaming behaviour assumed.")
    if p["gravity"].startswith("1") and f["kind"] == "liquid":
        notes.append("1 g selected: droplet burning rate is the same textbook law; only the flame shape differs on Earth.")
    if f["kind"] == "solid":
        times, m, panel, info = solid_fire(f, p, smoulder, notes)
    else:
        times, m, panel, info = liquid_fire(f, p, notes)

    eta = ETA_SMOULDER if smoulder else ETA_FLAMING
    smoke_yield = f["soot"] * (5 if smoulder else 1)
    hrr = {r: m[r] * f["LHV"] * eta / 1000 for r in RUNS}                     # kW
    integ = lambda y: float(np.sum((y[1:] + y[:-1]) / 2 * np.diff(times)))
    air_mass = RHO_AIR * CABIN["lx"] * CABIN["ly"] * CABIN["depth"]
    summary = {}
    for r in RUNS:
        mass = integ(m[r]); energy = mass * f["LHV"] * eta
        d_x = (mass * f["s"] / 0.032) / (air_mass / 0.02897) * 100              # O2 drop, percentage points
        alive = np.nonzero(m[r] > 0)[0]
        summary[r] = dict(peak_hrr_kW=sig(float(hrr[r].max())), time_of_peak_s=sig(float(times[int(np.argmax(m[r]))])),
                          fire_duration_s=sig(float(times[alive[-1]]) if len(alive) else 0.0),
                          fuel_burned_g=sig(mass * 1000), energy_released_kJ=sig(energy / 1000),
                          cabin_O2_drop_pct_points=sig(d_x),
                          cabin_temp_rise_upper_bound_K=sig(energy / (air_mass * CP_AIR)))

    rate_mg = m["mid"] * smoke_yield * 1e6                                     # mg/s
    rate_fn = lambda t: float(np.interp(t, times, rate_mg, right=0.0))
    flow_air = max(p["flow"], 1.0) / 100
    smoke = smoke_run(rate_fn, (p["fire_x"], p["fire_y"]), flow_air)
    names = list(smoke["alarms"])
    ok = {k: v for k, v in smoke["alarms"].items() if v is not None}
    smoke_out = dict(
        smoke_release_peak_mg_s=sig(float(rate_mg.max())), smoke_yield_kg_per_kg=sig(smoke_yield),
        alarm_threshold_mg_m3=ALARM_MG_M3, detector_delay_s=sig(smoke["delay"]), window_s=int(smoke["times"][-1]),
        alarm_time_s={k: (sig(v) if v is not None else None) for k, v in smoke["alarms"].items()},
        peak_concentration_mg_m3={k: sig(v) for k, v in smoke["peaks"].items()},
        first_to_alarm=(min(ok, key=ok.get) if ok else None), calibration=smoke["cal"])
    if len(ok) == 2:
        a, b = names
        smoke_out["ceiling_vs_return_gap_s"] = sig(abs(ok[a] - ok[b]))
    result = dict(
        inputs=dict(fuel=p["fuel"], flame_volume_cm3=p["volume_cm3"], oxygen_pct=p["o2"], airflow_cm_s=p["flow"],
                    gravity=p["gravity"], smoke_sensor_flame_type=p["smoke_type"],
                    flame_position_m=[p["fire_x"], p["fire_y"], p["fire_z"]], distance_m=p["distance"],
                    cabin=f"{CABIN['lx']} x {CABIN['ly']} x {CABIN['depth']} m"),
        fire=info, fire_summary=summary, smoke=smoke_out, notes=notes,
        _plot=dict(times=times, hrr=hrr, panel=panel, smoke=smoke))
    return result


# ------------------------------------------------------------------ LLM report (SARA style)
def llm_report(result):
    import groq
    client = groq.Client(api_key=GROQ_API_KEY)
    data = json.dumps({k: v for k, v in result.items() if not k.startswith("_")}, indent=1)
    last = None
    for model in MODELS:
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0.5, max_tokens=4000,
                messages=[{"role": "system", "content": SYSTEM_PROMPT},
                          {"role": "user", "content": "Simulation output:\n" + data + "\n\nWrite the risk assessment report."}])
            text = resp.choices[0].message.content
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            text = re.sub(r"[*#`]", "", text).strip()
            if text:
                return text, model
        except Exception as e:
            last = e
    raise RuntimeError(f"LLM call failed: {last}")


def speak(text):
    """Same approach as SARA: edge-tts to mp3, play with mpg123."""
    try:
        import edge_tts
        async def go():
            await edge_tts.Communicate(text, VOICE).save("report_audio.mp3")
        asyncio.run(go())
        os.system("mpg123 -q report_audio.mp3")
        os.remove("report_audio.mp3")
    except Exception as e:
        print("speech failed:", e)


# ------------------------------------------------------------------ UI
def main():
    import tkinter as tk
    from tkinter import ttk, filedialog, messagebox
    import matplotlib
    matplotlib.use("TkAgg")
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    root = tk.Tk(); root.title("SARA Fire Risk Assessor"); root.geometry("1250x760")
    root.columnconfigure(1, weight=1); root.rowconfigure(0, weight=1)
    q = queue.Queue(); last = {}

    left = ttk.Frame(root, padding=10); left.grid(row=0, column=0, sticky="ns")
    vars_ = {}

    def field(label, key, default, row):
        ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=2)
        v = tk.StringVar(value=str(default)); vars_[key] = v
        ttk.Entry(left, textvariable=v, width=12).grid(row=row, column=1, sticky="e", pady=2)

    def combo(label, key, values, default, row):
        ttk.Label(left, text=label).grid(row=row, column=0, sticky="w", pady=2)
        v = tk.StringVar(value=default); vars_[key] = v
        c = ttk.Combobox(left, textvariable=v, values=values, state="readonly", width=20)
        c.grid(row=row, column=1, sticky="e", pady=2); return c

    r = 0
    ttk.Label(left, text="FLAME (from detection system)", font=("TkDefaultFont", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w"); r += 1
    field("Flame volume (cm³)", "volume_cm3", 50, r); r += 1
    field("Position X in cabin (m, 0-4)", "fire_x", 0.6, r); r += 1
    field("Position Y in cabin (m, 0-2)", "fire_y", 1.4, r); r += 1
    field("Position Z (m)", "fire_z", 1.0, r); r += 1
    field("Distance from dome (m)", "distance", 1.0, r); r += 1
    ttk.Button(left, text="Load flame_state.json", command=lambda: load_json()).grid(row=r, column=0, columnspan=2, sticky="ew", pady=4); r += 1

    ttk.Label(left, text="FUEL", font=("TkDefaultFont", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(8, 0)); r += 1
    fuel_cb = combo("Fuel", "fuel", list(FUELS), list(FUELS)[0], r); r += 1
    field("Thickness (mm, solids)", "thick_mm", FUELS[list(FUELS)[0]]["thick_mm"], r); r += 1
    field("Fuel patch length (cm)", "patch_L", 20, r); r += 1
    field("Fuel patch width (cm)", "patch_W", 10, r); r += 1

    ttk.Label(left, text="SMOKE SENSOR + CABIN", font=("TkDefaultFont", 10, "bold")).grid(row=r, column=0, columnspan=2, sticky="w", pady=(8, 0)); r += 1
    combo("Sensor flame type", "smoke_type", SMOKE_TYPES, "Flaming", r); r += 1
    field("Oxygen (%)", "o2", 21, r); r += 1
    field("Airflow (cm/s)", "flow", 10, r); r += 1
    combo("Gravity", "gravity", ["0 g (orbit)", "1 g (Earth)"], "0 g (orbit)", r); r += 1

    run_btn = ttk.Button(left, text="RUN SIMULATION + REPORT", command=lambda: start())
    run_btn.grid(row=r, column=0, columnspan=2, sticky="ew", pady=(12, 2)); r += 1
    speak_btn = ttk.Button(left, text="Read report aloud", command=lambda: threading.Thread(
        target=speak, args=(txt.get("1.0", "end").strip(),), daemon=True).start())
    speak_btn.grid(row=r, column=0, columnspan=2, sticky="ew", pady=2); r += 1
    status = tk.StringVar(value="Ready"); ttk.Label(left, textvariable=status, wraplength=250, foreground="#555").grid(row=r, column=0, columnspan=2, sticky="w", pady=6)

    def on_fuel(_=None):
        f = FUELS[vars_["fuel"].get()]
        if f["kind"] == "solid":
            vars_["thick_mm"].set(f["thick_mm"])
    fuel_cb.bind("<<ComboboxSelected>>", on_fuel)

    nb = ttk.Notebook(root); nb.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
    t1 = ttk.Frame(nb); t2 = ttk.Frame(nb); t3 = ttk.Frame(nb)
    nb.add(t1, text="Risk report"); nb.add(t2, text="Plots"); nb.add(t3, text="Raw simulation output")
    txt = tk.Text(t1, wrap="word", font=("TkDefaultFont", 11), padx=10, pady=10)
    sb = ttk.Scrollbar(t1, command=txt.yview); txt.configure(yscrollcommand=sb.set)
    sb.pack(side="right", fill="y"); txt.pack(fill="both", expand=True)
    raw = tk.Text(t3, wrap="none", font=("TkFixedFont", 9))
    raw.pack(fill="both", expand=True)
    fig = Figure(figsize=(9, 6), dpi=100); canvas = FigureCanvasTkAgg(fig, master=t2)
    canvas.get_tk_widget().pack(fill="both", expand=True)

    def load_json(path=None):
        path = path or filedialog.askopenfilename(initialdir=HERE, filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            d = json.loads(Path(path).read_text())
            for key, names in (("volume_cm3", ("volume_cm3", "volume")), ("fire_x", ("x_m", "x")), ("fire_y", ("y_m", "y")),
                               ("fire_z", ("z_m", "z")), ("distance", ("distance_m", "distance"))):
                for n in names:
                    if n in d:
                        vars_[key].set(d[n]); break
            status.set(f"Loaded {Path(path).name}")
        except Exception as e:
            messagebox.showerror("Load failed", str(e))

    if (HERE / "flame_state.json").exists():
        load_json(HERE / "flame_state.json")

    def read_inputs():
        p = {k: v.get() for k, v in vars_.items()}
        for k in ("volume_cm3", "fire_x", "fire_y", "fire_z", "distance", "thick_mm", "patch_L", "patch_W", "o2", "flow"):
            p[k] = float(p[k])
        if p["volume_cm3"] <= 0 or p["patch_L"] <= 0 or p["patch_W"] <= 0 or p["thick_mm"] <= 0:
            raise ValueError("volume, thickness and patch size must be above zero")
        if not 5 <= p["o2"] <= 40:
            raise ValueError("oxygen should be between 5 and 40 %")
        if p["flow"] < 0:
            raise ValueError("airflow cannot be negative")
        return p

    def start():
        try:
            p = read_inputs()
        except Exception as e:
            messagebox.showerror("Check inputs", str(e)); return
        run_btn.state(["disabled"]); txt.delete("1.0", "end"); status.set("Running fire and smoke simulations...")
        threading.Thread(target=work, args=(p,), daemon=True).start()

    def work(p):
        try:
            res = run_all(p); q.put(("sim", res))
            q.put(("status", "Asking SARA for the risk report..."))
            text, model = llm_report(res)
            q.put(("report", (text, model)))
        except Exception as e:
            q.put(("error", str(e)))

    def draw(res):
        P = res["_plot"]; fig.clear(); S = P["smoke"]
        ax = fig.subplots(2, 3)
        cols = {"lo": "tab:blue", "mid": "k", "hi": "tab:red"}
        for rn in RUNS:
            ax[0][0].plot(P["times"], P["hrr"][rn], color=cols[rn], label=rn)
            ax[0][1].plot(P["times"], P["panel"][2][rn], color=cols[rn], label=rn)
        ax[0][0].set_title("Heat release (kW)", fontsize=9); ax[0][0].set_xlabel("s"); ax[0][0].legend(fontsize=7)
        ax[0][1].set_title(P["panel"][0] + f" ({P['panel'][1]})", fontsize=9); ax[0][1].set_xlabel("s")
        for name, s in S["probes"].items():
            ax[0][2].plot(S["times"], s, label=name.split(",")[0])
        ax[0][2].axhline(ALARM_MG_M3, color="grey", ls="--", lw=.8)
        ax[0][2].set_title("Smoke at detectors (mg/m³)", fontsize=9); ax[0][2].set_xlabel("s"); ax[0][2].legend(fontsize=7)
        ft = sorted(S["frames"])[:3]
        for a, t in zip(ax[1], ft):
            a.imshow(np.log10(1 + S["frames"][t]), origin="lower", extent=[0, CABIN["lx"], 0, CABIN["ly"]], cmap="Blues")
            a.plot((S["fire_cell"][0] + .5) * DX, (S["fire_cell"][1] + .5) * DX, "o", color="#d9480f")
            for nm, (i, j) in DETECTORS.items():
                a.plot((i + .5) * DX, (j + .5) * DX, "s", color="k", ms=4)
            a.set_title(f"Smoke map t = {t} s", fontsize=9)
        for a in ax[1][len(ft):]:
            a.axis("off")
        fig.tight_layout(); canvas.draw()

    def poll():
        try:
            while True:
                kind, val = q.get_nowait()
                if kind == "status":
                    status.set(val)
                elif kind == "sim":
                    last["res"] = val; draw(val)
                    raw.delete("1.0", "end")
                    raw.insert("end", json.dumps({k: v for k, v in val.items() if not k.startswith("_")}, indent=1))
                    txt.insert("end", "Simulations done. Waiting for the risk report...\n")
                elif kind == "report":
                    text, model = val
                    txt.delete("1.0", "end"); txt.insert("end", text)
                    status.set(f"Done (report by {model})"); run_btn.state(["!disabled"])
                elif kind == "error":
                    txt.delete("1.0", "end")
                    txt.insert("end", "Report generation failed:\n" + val + "\n\nThe simulation results are still in the Plots and Raw tabs.")
                    status.set("Error"); run_btn.state(["!disabled"])
        except queue.Empty:
            pass
        root.after(100, poll)

    poll(); root.mainloop()


if __name__ == "__main__":
    main()